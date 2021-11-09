######################################################################

# pool_size is given when instance_size and instance_stride are none. In this case, 2D ave_pool is applied to the feature extracted from the entire bag image

# On contrary, if instance_size and instance_stride are given, pool_size should be none, Once features are extracted from the instance, global ave_pool is applied

######################################################################

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D
from keras.utils import print_summary

def load_image_list( out_dir ):

    img_files = []
    fd = open( out_dir + 'sample_images.csv' )
    for line in fd:
        files = [ fn.strip() for fn in line.split(',')[1:] if fn.strip() != '' ]
        img_files.extend( files )
    return img_files


src_dir  = './data/HBS_cropped_images/'

if len(src_dir) > 1 and src_dir[-1] != '/':
    src_dir += '/'
    
out_dir = './data/HBS_cropped_images/cnn_feature'
if len(out_dir) > 1 and out_dir[-1] != '/':
    out_dir += '/'
model_name = 'vgg16'
layer = 'block4_pool'
list_layers = True
instance_size = 128
instance_stride = 128
pool_size = None

#load filenames and labels
image_list = load_image_list( out_dir )

# create model
max_dim = None
input_tensor = Input(shape=(max_dim,max_dim,3))
if model_name.lower() == 'vgg16':
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg16 import preprocess_input
    base_model = VGG16(input_shape=(max_dim,max_dim,3),include_top=False,weights='imagenet')

x = base_model.get_layer(layer).output #extract output of the 4th convolutional layer

# for creating instances(start here)
if pool_size is None and instance_size is None:
    x = GlobalAveragePooling2D(name='avgpool')(x)
elif pool_size is not None:
    p = int(pool_size)
    if p > 0:
        x = AveragePooling2D((p,p),name='avgpool')(x)
elif instance_size is not None:
    size = int(instance_size)
    if instance_stride is not None:
        stride = int(instance_stride)
    else:
        stride = size
    x = GlobalAveragePooling2D(name='avgpool')(x)
model = Model( inputs=base_model.input, outputs=x )

#create batch
for img_fn in image_list:

    img = image.load_img( src_dir+img_fn )
    x = image.img_to_array( img )
    x = np.expand_dims( x, axis=0 )
    x = preprocess_input( x )

    if instance_size is not None:
        feat = []
        bs = 16 #batch size
        x_batch = []
        for r in range(0,x.shape[1],stride):
            for c in range(0,x.shape[2],stride):
                x_inst = x[:,r:min(r+size,x.shape[1]),c:min(c+size,x.shape[2]),:]# x=image, size = instance_size, r = increment by stride size
                
                if len(x_batch) >= bs or ( len(x_batch) > 0 and x_inst.shape != x_batch[0].shape ):
                    # process a batch
                    if len(x_batch) > 1:
                        x_batch = np.concatenate(x_batch,axis=0)
                    else:
                        x_batch = x_batch[0]
                    feat_batch = model.predict(x_batch)
                    feat.append( feat_batch )
                    x_batch = []
                x_batch.append( x_inst )
        if len(x_batch) > 0:
            # process last batch
            if len(x_batch) > 1:
                x_batch = np.concatenate(x_batch,axis=0)
            else:
                x_batch = x_batch[0]
            feat_batch = model.predict(x_batch)
            feat.append( feat_batch )
        if len(feat) > 0:
            feat = np.concatenate( feat, axis=0 )
            feat = [ feat[r,:] for r in range(feat.shape[0]) ]

    #instance size is not known
    else:
        p = model.predict(x)
        if len(p.shape) > 2:
            feat = [ p[:,r,c,:].squeeze() for r in range(p.shape[1]) for c in range(p.shape[2]) ]
        else:
            feat = [ p.squeeze() ]
    if len(feat) > 0:
        print('%d x %d' % (len(feat),feat[0].shape[0]))
    else:
        print('no instances')

    feat_fn = out_dir+img_fn[:img_fn.rfind('.')]+'_'+model_name+'-'+layer
    if pool_size is not None:
        feat_fn += '_p'+str(pool_size)
    if instance_size is not None:
        feat_fn += '_i'+str(instance_size)
        if instance_stride is not None:
            feat_fn += '-'+str(instance_stride)
    feat_fn += '.npy'
    print('Saving '+feat_fn)

    if not os.path.exists( os.path.dirname(feat_fn) ):
        os.makedirs( os.path.dirname(feat_fn) )
    np.save(feat_fn,feat)

