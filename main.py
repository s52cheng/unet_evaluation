# from model import *
# from data import *
# from keras.callbacks import ModelCheckpoint  # Add this import

# #os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# data_gen_args = dict(rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05, 
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')
# myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

# model = unet()
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
# model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])
 
# testGene = testGenerator("data/membrane/test")
# results = model.predict_generator(testGene,30,verbose=1)
# saveResult("data/membrane/test",results)


from model import *
from data import *
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def pixel_accuracy(y_true, y_pred):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    y_pred_bin = tf.cast(y_pred_f > 0.5, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true_f, y_pred_bin), tf.float32))

# Training（if the model is trained then skip）
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir=None)
model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.keras', monitor='loss', verbose=1, save_best_only=True)
model.fit(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])

# 评估
testGene = testGenerator("data/membrane/train", num_image=30)  # 用训练集
predictions = []
ground_truths = []
steps = 30  # 训练集图片数量

for i in range(steps):
    img, mask = next(testGene)
    pred = model.predict(img)
    predictions.append(pred)
    ground_truths.append(mask)

predictions = np.concatenate(predictions, axis=0)
ground_truths = np.concatenate(ground_truths, axis=0)

from skimage.metrics import adapted_rand_error

# predictions and ground_truths should be numpy arrays of shape (N, H, W, 1)
dice_scores = []
pixel_errors = []
rand_errors = []
for y_true, y_pred in zip(ground_truths, predictions):
    y_pred_bin = (y_pred > 0.5).astype(np.float32)
    y_true_bin = (y_true > 0.5).astype(np.float32)
    # Dice
    dice = dice_coef(y_true_bin, y_pred_bin).numpy()
    dice_scores.append(dice)
    # Pixel error
    pixel_error = 1 - pixel_accuracy(y_true_bin, y_pred_bin).numpy()
    pixel_errors.append(pixel_error)
    # Rand error
    are, _, _ = adapted_rand_error(
        y_true_bin.squeeze().astype(np.uint8),
        y_pred_bin.squeeze().astype(np.uint8)
    )
    rand_errors.append(are)
print("Mean Dice coefficient:", np.mean(dice_scores))
print("Mean Pixel error:", np.mean(pixel_errors))
print("Mean Rand error:", np.mean(rand_errors)) 