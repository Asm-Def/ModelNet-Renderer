import os, shutil, glob
classnames = ['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
train_meshes = 'G:/JasonWu/ModelNet/ModelNet40/{}/train'
test_meshes = 'G:/JasonWu/ModelNet/ModelNet40/{}/test'
train_img = 'data/train'
test_img = 'data/test'

for classname in classnames:
    test_path = test_meshes.format(classname)
    names = glob.glob(os.path.join(test_path, '*.off'))
    if not os.path.exists(os.path.join(test_img, classname)):
        os.mkdir(os.path.join(test_img, classname))
    for name in names:
        fname = os.path.split(name)[-1].split('.')[0]
        testnames = glob.glob(os.path.join(train_img, classname, fname + '_*.png'))
        for testname in testnames:
            tname = os.path.split(testname)[-1]
            print(testname, os.path.join(test_img, classname, tname))
            shutil.move(testname, os.path.join(test_img, classname, tname))

