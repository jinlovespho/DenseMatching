import glob
import os


def verify_basesets(img_ann_path, img_jpg_path, seg_ann_path):
    print('1. Verifying ImageAnnotation, JPEGImages, Segmentation directories:')

    img_ann_catpaths = glob.glob('%s/*' % img_ann_path)
    img_jpg_catpaths = glob.glob('%s/*' % img_jpg_path)
    seg_ann_catpaths = glob.glob('%s/*' % seg_ann_path)

    img_ann_catpaths.sort()
    img_jpg_catpaths.sort()
    seg_ann_catpaths.sort()

    assert len(img_ann_catpaths) == len(img_jpg_catpaths) == len(seg_ann_catpaths), 'The number of category NOT same.'

    for img_ann_catpath, img_jpg_catpath, seg_ann_catpath in zip(img_ann_catpaths, img_jpg_catpaths, seg_ann_catpaths):
        img_ann_category = os.path.basename(img_ann_catpath)
        img_jpg_category = os.path.basename(img_jpg_catpath)
        seg_ann_category = os.path.basename(seg_ann_catpath)

        assert img_ann_category == img_jpg_category == seg_ann_category, 'Categories not matching.'
        category = img_ann_category

        print('\tChecking \'%s\' category...' % category)

        img_anns = glob.glob('%s/*.json' % img_ann_catpath)
        img_jpgs = glob.glob('%s/*.jpg' % img_jpg_catpath)
        seg_anns = glob.glob('%s/*.png' % seg_ann_catpath)

        img_anns.sort()
        img_jpgs.sort()
        seg_anns.sort()

        assert len(img_anns) == len(img_jpgs) == len(seg_anns) == 100, 'The number of files in %s NOT same' % category

        for img_ann, img_jpg, seg_ann in zip(img_anns, img_jpgs, seg_anns):
            img_ann_id = os.path.basename(img_ann).split('.')[0]
            img_jpg_id = os.path.basename(img_jpg).split('.')[0]
            seg_ann_id = os.path.basename(seg_ann).split('.')[0]

            assert img_ann_id == img_jpg_id == seg_ann_id, 'File ids not matching.'

    return True


def verify_splits(pairann_path, layout_path):
    print('2. Verifying SameSet(PairAnnnotation, Layout_Large), SubSet(Layout_Large > Layout_Small):')

    trn_pairann_path = os.path.join(pairann_path, 'trn')
    val_pairann_path = os.path.join(pairann_path, 'val')
    test_pairann_path = os.path.join(pairann_path, 'test')

    trn_pairanns = glob.glob('%s/*.json' % trn_pairann_path)
    val_pairanns = glob.glob('%s/*.json' % val_pairann_path)
    test_pairanns = glob.glob('%s/*.json' % test_pairann_path)

    trn_pairanns.sort()
    val_pairanns.sort()
    test_pairanns.sort()

    trn_layoutL_path = os.path.join(layout_path, 'large/trn.txt')
    val_layoutL_path = os.path.join(layout_path, 'large/val.txt')
    test_layoutL_path = os.path.join(layout_path, 'large/test.txt')

    trn_layoutL_file = open(trn_layoutL_path).read().split('\n')
    val_layoutL_file = open(val_layoutL_path).read().split('\n')
    test_layoutL_file = open(test_layoutL_path).read().split('\n')

    trn_layoutL_lines = trn_layoutL_file[:len(trn_layoutL_file) - 1]
    val_layoutL_lines = val_layoutL_file[:len(val_layoutL_file) - 1]
    test_layoutL_lines = test_layoutL_file[:len(test_layoutL_file) - 1]

    trn_layoutL_lines.sort()
    val_layoutL_lines.sort()
    test_layoutL_lines.sort()

    assert len(trn_pairanns) == len(trn_layoutL_lines) and \
           len(val_pairanns) == len(val_layoutL_lines) and \
           len(test_pairanns) == len(test_layoutL_lines), \
        'The size of PairAnnotation splits and Layout(large) splits NOT same.'

    print('\tChecking SameSet(PairAnnnotation/trn, Layout/large/trn.txt)...')
    for trn_pairann, trn_layoutL_line in zip(trn_pairanns, trn_layoutL_lines):
        assert os.path.basename(trn_pairann).split('.')[0] == trn_layoutL_line, 'Pair ids not matching.'

    print('\tChecking SameSet(PairAnnnotation/val, Layout/large/val.txt)...')
    for val_pairann, val_layoutL_line in zip(val_pairanns, val_layoutL_lines):
        assert os.path.basename(val_pairann).split('.')[0] == val_layoutL_line, 'Pair ids not matching.'

    print('\tChecking SameSet(PairAnnnotation/test, Layout/large/test.txt)...')
    for test_pairann, test_layoutL_line in zip(test_pairanns, test_layoutL_lines):
        assert os.path.basename(test_pairann).split('.')[0] == test_layoutL_line, 'Pair ids not matching.'


    trn_layoutL_lines = dict(zip(trn_layoutL_lines, trn_layoutL_lines))
    val_layoutL_lines = dict(zip(val_layoutL_lines, val_layoutL_lines))
    test_layoutL_lines = dict(zip(test_layoutL_lines, test_layoutL_lines))

    trn_layoutS_path = os.path.join(layout_path, 'small/trn.txt')
    val_layoutS_path = os.path.join(layout_path, 'small/val.txt')
    test_layoutS_path = os.path.join(layout_path, 'small/test.txt')

    trn_layoutS_file = open(trn_layoutS_path).read().split('\n')
    val_layoutS_file = open(val_layoutS_path).read().split('\n')
    test_layoutS_file = open(test_layoutS_path).read().split('\n')

    trn_layoutS_lines = trn_layoutS_file[:len(trn_layoutS_file) - 1]
    val_layoutS_lines = val_layoutS_file[:len(val_layoutS_file) - 1]
    test_layoutS_lines = test_layoutS_file[:len(test_layoutS_file) - 1]

    trn_layoutS_lines.sort()
    val_layoutS_lines.sort()
    test_layoutS_lines.sort()

    assert len(trn_layoutL_lines) > len(trn_layoutS_lines) and \
           len(val_layoutL_lines) > len(val_layoutS_lines) and \
           len(test_layoutL_lines) > len(test_layoutS_lines), \
        'Large splits are smaller than Small splits.'

    print('\tChecking Subset(Layout/large/trn.txt > Layout/small/trn.txt)...')
    for trn_layoutS_line in trn_layoutS_lines:
        assert trn_layoutL_lines.get(trn_layoutS_line) is not None, 'Small layout is not in Large layout.'

    print('\tChecking Subset(Layout/large/val.txt > Layout/small/val.txt)...')
    for val_layoutS_line in val_layoutS_lines:
        assert val_layoutL_lines.get(val_layoutS_line) is not None, 'Small layout is not in Large layout.'

    print('\tChecking Subset(Layout/large/test.txt > Layout/small/test.txt)...')
    for test_layoutS_line in test_layoutS_lines:
        assert test_layoutL_lines.get(test_layoutS_line) is not None, 'Small layout is not in Large layout.'

    return True


def verify_disjointedness(layout_path):
    print('3. Verifying disjointedness between trn, val, test pairs:')

    trn_layoutL_path = os.path.join(layout_path, 'large/trn.txt')
    val_layoutL_path = os.path.join(layout_path, 'large/val.txt')
    test_layoutL_path = os.path.join(layout_path, 'large/test.txt')
    trn_layoutS_path = os.path.join(layout_path, 'small/trn.txt')
    val_layoutS_path = os.path.join(layout_path, 'small/val.txt')
    test_layoutS_path = os.path.join(layout_path, 'small/test.txt')

    trn_layoutL_file = open(trn_layoutL_path).read().split('\n')
    val_layoutL_file = open(val_layoutL_path).read().split('\n')
    test_layoutL_file = open(test_layoutL_path).read().split('\n')
    trn_layoutS_file = open(trn_layoutS_path).read().split('\n')
    val_layoutS_file = open(val_layoutS_path).read().split('\n')
    test_layoutS_file = open(test_layoutS_path).read().split('\n')

    trn_layoutL_lines = trn_layoutL_file[:len(trn_layoutL_file) - 1]
    val_layoutL_lines = val_layoutL_file[:len(val_layoutL_file) - 1]
    test_layoutL_lines = test_layoutL_file[:len(test_layoutL_file) - 1]
    trn_layoutS_lines = trn_layoutS_file[:len(trn_layoutS_file) - 1]
    val_layoutS_lines = val_layoutS_file[:len(val_layoutS_file) - 1]
    test_layoutS_lines = test_layoutS_file[:len(test_layoutS_file) - 1]

    assert len(set(trn_layoutL_lines)) == len(trn_layoutL_lines) and \
           len(set(val_layoutL_lines)) == len(val_layoutL_lines) and \
           len(set(test_layoutL_lines)) == len(test_layoutL_lines), 'Same pair exists within single split (Large).'
    assert len(set(trn_layoutS_lines)) == len(trn_layoutS_lines) and \
           len(set(val_layoutS_lines)) == len(val_layoutS_lines) and \
           len(set(test_layoutS_lines)) == len(test_layoutS_lines), 'Same pair exists within single split (Small).'

    trn_layoutL_imgs = dict()
    val_layoutL_imgs = dict()
    test_layoutL_imgs = dict()

    for trn_layoutL_line in trn_layoutL_lines:
        cat = trn_layoutL_line.split(':')[1]
        trn_layoutL_line = trn_layoutL_line.split(':')[0]
        src_img = trn_layoutL_line.split('-')[1]
        trg_img = trn_layoutL_line.split('-')[2]
        trn_layoutL_imgs[src_img] = cat
        trn_layoutL_imgs[trg_img] = cat

    for val_layoutL_line in val_layoutL_lines:
        cat = val_layoutL_line.split(':')[1]
        val_layoutL_line = val_layoutL_line.split(':')[0]
        src_img = val_layoutL_line.split('-')[1]
        trg_img = val_layoutL_line.split('-')[2]
        val_layoutL_imgs[src_img] = cat
        val_layoutL_imgs[trg_img] = cat

    for test_layoutL_line in test_layoutL_lines:
        cat = test_layoutL_line.split(':')[1]
        test_layoutL_line = test_layoutL_line.split(':')[0]
        src_img = test_layoutL_line.split('-')[1]
        trg_img = test_layoutL_line.split('-')[2]
        test_layoutL_imgs[src_img] = cat
        test_layoutL_imgs[trg_img] = cat

    print('\tChecking if disjoint(TrnImg, ValImg, TestImg) in Large...')
    for test_layoutL_img in test_layoutL_imgs:
        if trn_layoutL_imgs.get(test_layoutL_img) is not None:
            assert test_layoutL_imgs[test_layoutL_img] != trn_layoutL_imgs.get(test_layoutL_img), \
                'Test image is in Train image (Large).'
        if val_layoutL_imgs.get(test_layoutL_img) is not None:
            assert test_layoutL_imgs[test_layoutL_img] != val_layoutL_imgs.get(test_layoutL_img), \
                'Test image is in Validation image (Large).'

    for val_layoutL_img in val_layoutL_imgs:
        if trn_layoutL_imgs.get(val_layoutL_img) is not None:
            assert val_layoutL_imgs[val_layoutL_img] != trn_layoutL_imgs.get(val_layoutL_img), \
                'Val image is in Train image (Large).'

    trn_layoutS_imgs = dict()
    val_layoutS_imgs = dict()
    test_layoutS_imgs = dict()

    for trn_layoutS_line in trn_layoutS_lines:
        cat = trn_layoutS_line.split(':')[1]
        trn_layoutS_line = trn_layoutS_line.split(':')[0]
        src_img = trn_layoutS_line.split('-')[1]
        trg_img = trn_layoutS_line.split('-')[2]
        trn_layoutS_imgs[src_img] = cat
        trn_layoutS_imgs[trg_img] = cat

    for val_layoutS_line in val_layoutS_lines:
        cat = val_layoutS_line.split(':')[1]
        val_layoutS_line = val_layoutS_line.split(':')[0]
        src_img = val_layoutS_line.split('-')[1]
        trg_img = val_layoutS_line.split('-')[2]
        val_layoutS_imgs[src_img] = cat
        val_layoutS_imgs[trg_img] = cat

    for test_layoutS_line in test_layoutS_lines:
        cat = test_layoutS_line.split(':')[1]
        test_layoutS_line = test_layoutS_line.split(':')[0]
        src_img = test_layoutS_line.split('-')[1]
        trg_img = test_layoutS_line.split('-')[2]
        test_layoutS_imgs[src_img] = cat
        test_layoutS_imgs[trg_img] = cat

    print('\tChecking if disjoint(TrnImg, ValImg, TestImg) in Small...')
    for test_layoutS_img in test_layoutS_imgs:
        if trn_layoutS_imgs.get(test_layoutS_img) is not None:
            assert test_layoutS_imgs[test_layoutS_img] != trn_layoutS_imgs.get(test_layoutS_img), \
                'Test image is in Train image (Small).'
        if val_layoutS_imgs.get(test_layoutS_img) is not None:
            assert test_layoutS_imgs[test_layoutS_img] != val_layoutS_imgs.get(test_layoutS_img), \
                'Test image is in Validation image (Small).'

    for val_layoutS_img in val_layoutS_imgs:
        if trn_layoutS_imgs.get(val_layoutS_img) is not None:
            assert val_layoutS_imgs[val_layoutS_img] != trn_layoutS_imgs.get(val_layoutS_img), \
                'Val image is in Train image (Small).'

    trn_layoutL_lines = list(map(lambda x: x[x.index('-') + 1:], trn_layoutL_lines))
    val_layoutL_lines = list(map(lambda x: x[x.index('-') + 1:], val_layoutL_lines))
    test_layoutL_lines = list(map(lambda x: x[x.index('-') + 1:], test_layoutL_lines))
    trn_layoutS_lines = list(map(lambda x: x[x.index('-') + 1:], trn_layoutS_lines))
    val_layoutS_lines = list(map(lambda x: x[x.index('-') + 1:], val_layoutS_lines))
    test_layoutS_lines = list(map(lambda x: x[x.index('-') + 1:], test_layoutS_lines))

    trn_layoutL_set_lines = dict(zip(trn_layoutL_lines, trn_layoutL_lines))
    val_layoutL_set_lines = dict(zip(val_layoutL_lines, val_layoutL_lines))
    test_layoutL_set_lines = dict(zip(test_layoutL_lines, test_layoutL_lines))
    trn_layoutS_set_lines = dict(zip(trn_layoutS_lines, trn_layoutS_lines))
    val_layoutS_set_lines = dict(zip(val_layoutS_lines, val_layoutS_lines))
    test_layoutS_set_lines = dict(zip(test_layoutS_lines, test_layoutS_lines))

    assert len(trn_layoutL_set_lines) == len(trn_layoutL_lines) and \
           len(val_layoutL_set_lines) == len(val_layoutL_lines) and \
           len(test_layoutL_set_lines) == len(test_layoutL_lines) and \
           len(trn_layoutS_set_lines) == len(trn_layoutS_lines) and \
           len(val_layoutS_set_lines) == len(val_layoutS_lines) and \
           len(test_layoutS_set_lines) == len(test_layoutS_lines), 'Implementation Error.'

    print('\tChecking if disjoint(TrnPair, ValPair, TestPair) in Large...')
    for test_layoutL_set_line in test_layoutL_set_lines:
        assert trn_layoutL_set_lines.get(test_layoutL_set_line) is None, 'Test pair is in Train set (Large).'
        assert val_layoutL_set_lines.get(test_layoutL_set_line) is None, 'Test pair is in Val set (Large).'

    for val_layoutL_set_line in val_layoutL_set_lines:
        assert test_layoutL_set_lines.get(val_layoutL_set_line) is None, 'Val pair is in Test set (Large).'

    print('\tChecking if disjoint(TrnPair, ValPair, TestPair) in Small...')
    for test_layoutS_set_line in test_layoutS_set_lines:
        assert trn_layoutS_set_lines.get(test_layoutS_set_line) is None, 'Test pair is in Train set (Small).'
        assert val_layoutS_set_lines.get(test_layoutS_set_line) is None, 'Test pair is in Val set (Small).'

    for val_layoutS_set_line in val_layoutS_set_lines:
        assert test_layoutS_set_lines.get(val_layoutS_set_line) is None, 'Val pair is in Test set (Small).'

    return True


if __name__ == '__main__':
    img_ann_path = '../ImageAnnotation'
    img_jpg_path = '../JPEGImages'
    seg_ann_path = '../Segmentation'
    pairann_path = '../PairAnnotation'
    layout_path = '../Layout'

    if verify_basesets(img_ann_path, img_jpg_path, seg_ann_path):
        print('  ** All files in ImageAnnotation, JPEGImages, Segmentation are intact.\n')
    if verify_splits(pairann_path, layout_path):
        print('  ** All Layout splits and PairAnnotation are intact.\n')
    if verify_disjointedness(layout_path):
        print('  ** All pairs and images in trn, val, test are disjoint.\n')

    print('All data in SPair-71k dataset verified.')
