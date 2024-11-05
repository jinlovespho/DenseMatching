# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# CroCo model during pretraining
# --------------------------------------------------------


import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True # for gpu >= Ampere and pytorch >= 1.12
from functools import partial

from models.croco.blocks import Block, DecoderBlock, PatchEmbed
from models.croco.pos_embed import get_2d_sincos_pos_embed, RoPE2D 
from models.croco.masking import RandomMask

from torchvision import transforms
from utils_flow.pixel_wise_mapping import warp
import torch.nn.functional as F
from einops import rearrange



class CroCoNet(nn.Module):

    def __init__(self,
                 img_size=224,           # input image size
                 patch_size=16,          # patch_size 
                 mask_ratio=0.9,         # ratios of masked tokens 
                 enc_embed_dim=768,      # encoder feature dimension
                 enc_depth=12,           # encoder depth 
                 enc_num_heads=12,       # encoder number of heads in the transformer block 
                 dec_embed_dim=512,      # decoder feature dimension 
                 dec_depth=8,            # decoder depth 
                 dec_num_heads=16,       # decoder number of heads in the transformer block 
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_im2_in_dec=True,   # whether to apply normalization of the 'memory' = (second image) in the decoder 
                 pos_embed='cosine',     # positional embedding (either cosine or RoPE100)
                 args=None,
                ):
                
        super(CroCoNet, self).__init__()

        # self.args = args 
        self.model = args.model 
        self.reciprocity = args.reciprocity
        self.output_ca_map = args.output_ca_map
        self.softmax_camap = args.softmax_camap
        self.img_size = img_size

        if self.model == 'croco_catseg':
            from models.croco.cats_swin_decoder import CATs_SWIN_Decoder
            self.cats_swin_decoder = CATs_SWIN_Decoder(feature_size=(img_size[0]//16), hyperpixel_ids = [i for i in range(0, 12)], args=args)

        elif self.model == '':
            pass
                
        # patch embeddings  (with initialization done as in MAE)
        self._set_patch_embed(img_size, patch_size, enc_embed_dim)

        # mask generations
        self._set_mask_generator(self.patch_embed.num_patches, mask_ratio)

        self.pos_embed = pos_embed
        if pos_embed=='cosine':
            # positional embedding of the encoder 
            enc_pos_embed = get_2d_sincos_pos_embed(enc_embed_dim, self.patch_embed.grid_size, n_cls_token=0)
            self.register_buffer('enc_pos_embed', torch.from_numpy(enc_pos_embed).float())
            # positional embedding of the decoder  
            dec_pos_embed = get_2d_sincos_pos_embed(dec_embed_dim, self.patch_embed.grid_size, n_cls_token=0)
            self.register_buffer('dec_pos_embed', torch.from_numpy(dec_pos_embed).float())
            # pos embedding in each block
            self.rope = None # nothing for cosine 
        elif pos_embed.startswith('RoPE'): # eg RoPE100 
            self.enc_pos_embed = None # nothing to add in the encoder with RoPE
            self.dec_pos_embed = None # nothing to add in the decoder with RoPE
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(pos_embed[len('RoPE'):])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError('Unknown pos_embed '+pos_embed)

        # transformer for the encoder 
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)
        
        # masked tokens 
        self._set_mask_token(dec_embed_dim)

        # decoder 
        self._set_decoder(enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec, self.softmax_camap)
        
        # prediction head 
        # self._set_prediction_head(dec_embed_dim, patch_size)
        
        # initializer weights
        self.initialize_weights()           

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_embed_dim)

    def _set_mask_generator(self, num_patches, mask_ratio):
        self.mask_generator = RandomMask(num_patches, mask_ratio)
        
    def _set_mask_token(self, dec_embed_dim):
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        
    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec, softmax_camap):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder 
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder 
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope, softmax_camap=softmax_camap)
            for i in range(dec_depth)])
        # final norm layer 
        self.dec_norm = norm_layer(dec_embed_dim)
        
    def _set_prediction_head(self, dec_embed_dim, patch_size):
         self.prediction_head = nn.Linear(dec_embed_dim, patch_size**2 * 3, bias=True)
        
        
    def initialize_weights(self):
        # patch embed 
        self.patch_embed._init_weights()
        # mask tokens
        if self.mask_token is not None: torch.nn.init.normal_(self.mask_token, std=.02)
        # linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _encode_image(self, image, do_mask=False, return_all_blocks=False):
        """
        image has B x 3 x img_size x img_size 
        do_mask: whether to perform masking or not
        return_all_blocks: if True, return the features at the end of every block 
                           instead of just the features from the last block (eg for some prediction heads)
        """
        # embed the image into patches  (x has size B x Npatches x C) 
        # and get position if each return patch (pos has size B x Npatches x 2)
        x, pos = self.patch_embed(image)              
        # add positional embedding without cls token  
        if self.enc_pos_embed is not None: 
            x = x + self.enc_pos_embed[None,...]
        # apply masking 
        B,N,C = x.size()
        if do_mask:
            masks = self.mask_generator(x)
            x = x[~masks].view(B, -1, C)
            posvis = pos[~masks].view(B, -1, 2)
        else:
            B,N,C = x.size()
            masks = torch.zeros((B,N), dtype=bool)
            posvis = pos
        # now apply the transformer encoder and normalization        
        if return_all_blocks:
            out = []
            for blk in self.enc_blocks:
                x = blk(x, posvis)
                out.append(x)
            out[-1] = self.enc_norm(out[-1])
            return out, pos, masks
        else:
            for blk in self.enc_blocks:
                x = blk(x, posvis)
            x = self.enc_norm(x)
            return x, pos, masks
 
    def _decoder(self, feat1, pos1, masks1, feat2, pos2, return_all_blocks=False):
        """
        return_all_blocks: if True, return the features at the end of every block 
                           instead of just the features from the last block (eg for some prediction heads)
                           
        masks1 can be None => assume image1 fully visible 
        """
        # encoder to decoder layer 
        visf1 = self.decoder_embed(feat1)
        f2 = self.decoder_embed(feat2)
        # append masked tokens to the sequence
        B,Nenc,C = visf1.size()
        if masks1 is None: # downstreams
            f1_ = visf1
        else: # pretraining 
            Ntotal = masks1.size(1)
            f1_ = self.mask_token.repeat(B, Ntotal, 1).to(dtype=visf1.dtype)
            f1_[~masks1] = visf1.view(B * Nenc, C)
        # add positional embedding
        if self.dec_pos_embed is not None:
            f1_ = f1_ + self.dec_pos_embed
            f2 = f2 + self.dec_pos_embed
        # apply Transformer blocks
        out = f1_
        out2 = f2 
        attn_maps = []
        if return_all_blocks:
            _out, out = out, []
            for blk in self.dec_blocks:
                _out, out2, attn_map = blk(_out, out2, pos1, pos2)
                out.append(_out)
                attn_maps.append(attn_map)
            out[-1] = self.dec_norm(out[-1])
        else:
            for blk in self.dec_blocks:
                out, out2, attn_map = blk(out, out2, pos1, pos2)
                attn_maps.append(attn_map)
            out = self.dec_norm(out)
            
        if self.output_ca_map:
            return out, attn_maps
        return out, None

    def patchify(self, imgs):
        """
        imgs: (B, 3, H, W)
        x: (B, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        
        return x

    def unpatchify(self, x, channels=3):
        """
        x: (N, L, patch_size**2 *channels)
        imgs: (N, 3, H, W)
        """
        patch_size = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], channels, h * patch_size, h * patch_size))
        return imgs

    def resize_flow(self, flow, size):
        flow = flow.clone()
        h_o, w_o = flow.size()[-2:]
        h, w = size[-2:]
        
        ratio_h = float(h - 1)/float(h_o - 1)
        ratio_w = float(w - 1)/float(w_o - 1)
        flow[:, 0, :, :] *= ratio_w
        flow[:, 1, :, :] *= ratio_h

        flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=True)
        return flow
    
    def tile_image(self, img, tile_shape=(128, 128)):
        """
        Arguments:
            img: tensor shape of (3, 512, 512)
            tile_shape: tuple
        """
        _, H, W = img.shape
        tile = rearrange(img, 'C (T1 H) (T2 W) -> (T1 T2) C H W', H=tile_shape[0], W=tile_shape[1])
        return tile

    def tile_to_image(self, tile):
        T = int((tile.shape[0]) ** 0.5)
        return rearrange(tile, '(T1 T2) C H W -> C (T1 H) (T2 W)', T1=T, T2=T)

    def estimate_flow(self, target_img, source_img):
        output = self.forward(source_img, target_img)
        if self.model == 'croco_catseg':
            flow_est = output[0]  # fine flow
        else:
            flow_est = output
        return flow_est

    def zoom_in_batch(self, src_img, trg_img, zoom_ratio=(2,3), optimize=False, homo_only=False, batch_size=24):
        flow_list = []
        uncertainty_list = []
        '''
            src_img: b 3 h w 일 때 -> tmp=src_img.split(1) -> b개의 3 h w 이미지가 나옴 -> 즉 len(tmp)=b
        '''
        for src, trg in zip(src_img.split(1), trg_img.split(1)):   
            if homo_only:
                flow, _ = self.estimate_flow_and_confidence_map_(
                    src, trg, inference_parameters={
                        'mask_type': 'cyclic_consistency_error_below_10',
                        'multi_stage_type': 'homography_only',
                        'min_nbr_points': 10000,
                    })
            elif optimize:
                flow, uncertainty = self.zoom_in_with_optimize(src, trg, zoom_ratio, max_num_iter=100, batch_size=batch_size)
            else:
                flow, uncertainty = self.zoom_fix_multiscale(src, trg, zoom_ratio, batch_size=batch_size)
            flow_list.append(flow)
            uncertainty_list.append(uncertainty)
        return torch.cat(flow_list, dim=0), torch.cat(uncertainty_list, dim=0)
    

    def zoom_fix_multiscale(self, src_img, trg_img, zoom_ratio_list=(3, 4, 5), batch_size=24):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        src_img = src_img.to(device)
        trg_img = trg_img.to(device)

        src_img_resized = transforms.functional.resize(src_img, size=self.img_size)
        trg_img_resized = transforms.functional.resize(trg_img, size=self.img_size)
        with torch.no_grad():
            est_flow = self.estimate_flow(src_img_resized, trg_img_resized)
            est_flow_rev = self.estimate_flow(trg_img_resized, src_img_resized)

        est_flow = self.resize_flow(est_flow, trg_img.shape)
        est_flow = self.resize_flow(est_flow, trg_img.shape)
        est_flow_rev = self.resize_flow(est_flow_rev, src_img.shape)
        warped_src = warp(src_img.float(), est_flow)
        warped_trg = warp(trg_img.float(), est_flow_rev)

        final_flow_list = []
        final_flow_rev_list = []

        final_flow_list.append(est_flow)
        final_flow_rev_list.append(est_flow_rev)

        for zoom_ratio in zoom_ratio_list:
            warped_src_resized = F.interpolate(warped_src.float(), size=self.img_size[0] * zoom_ratio, mode='bilinear', align_corners=True)
            warped_src_tile = self.tile_image(warped_src_resized[0], tile_shape=(self.img_size[0], self.img_size[0]))

            resized_trg = F.interpolate(trg_img.float(), size=self.img_size[0] * zoom_ratio, mode='bilinear', align_corners=True)
            trg_tile = self.tile_image(resized_trg[0], tile_shape=(self.img_size[0], self.img_size[0]))


            est_flow_tile_list = []
            for s, t in zip(warped_src_tile.split(batch_size), trg_tile.split(batch_size)):
                with torch.no_grad():
                    est_flow_tile = self.estimate_flow(s.to('cuda'), t.to('cuda'))
                    est_flow_tile_list.append(est_flow_tile)

            est_flow_tile = torch.cat(est_flow_tile_list, dim=0)
            est_flow_zoom = self.tile_to_image(est_flow_tile)

            est_flow_zoom_origsize = self.resize_flow(est_flow_zoom[None], trg_img.shape)
            warped_flow = warp(est_flow, est_flow_zoom_origsize)
            final_flow = warped_flow + est_flow_zoom_origsize

            final_flow_list.append(final_flow)

        for zoom_ratio in zoom_ratio_list:
            warped_trg_resized = F.interpolate(warped_trg.float(), size=self.img_size[0] * zoom_ratio, mode='bilinear', align_corners=True)
            warped_trg_tile = self.tile_image(warped_trg_resized[0], tile_shape=(self.img_size[0], self.img_size[0]))

            resized_src = F.interpolate(src_img.float(), size=self.img_size[0] * zoom_ratio, mode='bilinear', align_corners=True)
            src_tile = self.tile_image(resized_src[0], tile_shape=(self.img_size[0], self.img_size[0]))


            est_flow_tile_list = []
            for t, s in zip(warped_trg_tile.split(batch_size), src_tile.split(batch_size)):
                with torch.no_grad():
                    est_flow_tile = self.estimate_flow(t.to('cuda'), s.to('cuda'))
                    est_flow_tile_list.append(est_flow_tile)

            est_flow_tile = torch.cat(est_flow_tile_list, dim=0)
            est_flow_zoom = self.tile_to_image(est_flow_tile)

            est_flow_zoom_origsize = self.resize_flow(est_flow_zoom[None], src_img.shape)
            warped_flow = warp(est_flow_rev, est_flow_zoom_origsize)
            final_flow = warped_flow + est_flow_zoom_origsize

            final_flow_rev_list.append(final_flow)

        final_flow_list = torch.cat(final_flow_list, dim=0)
        final_flow_rev_list = torch.cat(final_flow_rev_list, dim=0)

        final_confidence_list = torch.norm(final_flow_list + warp(final_flow_rev_list, final_flow_list), dim=1, p=2, keepdim=True)
        final_confidence_list_rev = torch.norm(final_flow_rev_list + warp(final_flow_list, final_flow_rev_list), dim=1, p=2, keepdim=True)

        final_flow = torch.gather(final_flow_list, dim=0, index=final_confidence_list.min(dim=0, keepdim=True)[1].repeat(1, 2, 1, 1))
        final_flow_rev = torch.gather(final_flow_rev_list, dim=0, index=final_confidence_list_rev.min(dim=0, keepdim=True)[1].repeat(1, 2, 1, 1))

        return final_flow, torch.norm(final_flow + warp(final_flow_rev, final_flow), dim=1, p=2, keepdim=True)


    def forward(self, img_target, img_source, mode=None):
        """
        img1: tensor of size B x 3 x img_size x img_size
        img2: tensor of size B x 3 x img_size x img_size
        
        out will be    B x N x (3*patch_size*patch_size)
        masks are also returned as B x N just in case 
        """
        B,_,H,W = img_target.size()
        feat_targets, pos_target, mask_target = self._encode_image(img_target, do_mask=False, return_all_blocks=True)
        feat_sources, pos_source, mask_source = self._encode_image(img_source, do_mask=False, return_all_blocks=True)

        feat_target = feat_targets[-1]
        feat_source = feat_sources[-1]

        # decoder
        decfeat, attn_map = self._decoder(feat_target, pos_target, mask_target, feat_source, pos_source, return_all_blocks=True)
        if self.reciprocity:
            decfeat_source, attn_map_source = self._decoder(feat_source, pos_source, mask_source, feat_target, pos_target, return_all_blocks=True)
        
        ## heuristic attention refine
        attn_map = [attn.mean(dim=1).detach() for attn in attn_map]
        for i in range(len(attn_map)):
            attn_map[i][:,:,0]=attn_map[i].min()
        self.attn_map = attn_map
        if self.reciprocity:
            attn_map_source = [attn.mean(dim=1).detach() for attn in attn_map_source]
            for i in range(len(attn_map_source)):
                attn_map_source[i][:,:,0]=attn_map_source[i].min()    
        
        if self.model == 'cats_swin':
            decfeat = [feat.detach() for feat in decfeat]
            if self.reciprocity:
                decfeat_source = [feat.detach() for feat in decfeat_source]
                output_flow = self.cats_swin(attn_map, decfeat, (H,W), feat_source, feat_target, attn_map_source, decfeat_source, img_target, img_source)
            else:
                output_flow = self.cats_swin(attn_map, decfeat, (H,W), feat_source, feat_target)
            return output_flow
        
        elif self.model == 'croco_catseg':
            decfeat = [feat.detach() for feat in decfeat]
            if self.reciprocity:
                decfeat_source = [feat.detach() for feat in decfeat_source]
                output_flow = self.cats_swin_decoder(attn_map, decfeat, (H,W), feat_source, feat_target, attn_map_source, decfeat_source, img_target, img_source, appearance_feature = [feat_targets[8],feat_targets[16]])
                # output_flow = [fine_flow, coarse_flow]
            else:
                output_flow = self.cats_swin_decoder(attn_map, decfeat, (H,W), feat_source, feat_target, appearance_feature = [feat_targets[8],feat_targets[16]])
            return output_flow
        