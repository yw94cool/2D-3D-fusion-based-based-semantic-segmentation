from __future__ import print_function
import tensorflow as tf
import numpy as np

import resnet_model
import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange

import scipy.misc as misc
from matplotlib import pyplot as plt
from matplotlib import cm
import time
import pandas as pd
import glob
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "8", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string(
    "data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4",
                      "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "true", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 6
IMAGE_SIZE = 400


class ResModel(resnet_model.Model):
    """Model class with appropriate defaults for Imagenet data."""

    def __init__(self, resnet_size=50, data_format='channels_last', num_classes=1000,
                 version=1,
                 dtype=resnet_model.DEFAULT_DTYPE):
        """These are the parameters that work for Imagenet data.
        Args:
          resnet_size: The number of convolutional layers needed in the model.
          data_format: Either 'channels_first' or 'channels_last', specifying which
            data format to use when setting up the model.
          num_classes: The number of output classes needed from the model. This
            enables users to extend the same model to their own datasets.
          version: Integer representing which version of the ResNet network to use.
            See README for details. Valid values: [1, 2]
          dtype: The TensorFlow dtype to use for calculations.
        """

        # For bigger models, we want to use "bottleneck" layers
        if resnet_size < 50:
            bottleneck = False
            final_size = 512
        else:
            bottleneck = True
            final_size = 2048

        super(ResModel, self).__init__(
            resnet_size=resnet_size,
            bottleneck=bottleneck,
            num_classes=num_classes,
            num_filters=64,
            kernel_size=3,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            second_pool_size=7,
            second_pool_stride=1,
            block_sizes=_get_block_sizes(resnet_size),
            block_strides=[1, 1, 2, 2],
            final_size=final_size,
            version=version,
            data_format=data_format,
            dtype=dtype
        )

def _get_block_sizes(resnet_size):
    """Retrieve the size of each block_layer in the ResNet model.
    The number of block layers used for the Resnet model varies according
    to the size of the model. This helper grabs the layer set we want, throwing
    an error if a non-standard size has been selected.
    Args:
      resnet_size: The number of convolutional layers needed in the model.
    Returns:
      A list of block sizes to use in building the model.
    Raises:
      KeyError: if invalid resnet_size is received.
    """
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }

    try:
        return choices[resnet_size]
    except KeyError:
        err = ('Could not find layers for selected Resnet size.\n'
               'Size received: {}; sizes allowed: {}.'.format(
                   resnet_size, choices.keys()))
        raise ValueError(err)


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up ResNet50 backends for feature extraction ...")
    mean_pixel = [86.59, 92.48, 86.04, 90.07]
    processed_image = utils.process_image(image, mean_pixel)
 
    with tf.variable_scope("inference", reuse=tf.AUTO_REUSE):
        resnet = ResModel(resnet_size=34, data_format='channels_last')
        conv_final = resnet(processed_image, True)
        print('output shape of resnet', conv_final.get_shape())
        # get target shape of output
        target_shape = tf.stack([tf.shape(image)[0],IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CLASSESS])
        W_up = utils.weight_variable(
            [16, 16, NUM_OF_CLASSESS, conv_final.get_shape()[3].value], name="W_up")
        b_up = utils.bias_variable([NUM_OF_CLASSESS], name="b_up")
        conv_up = utils.conv2d_transpose_strided(
            conv_final, W_up, b_up, output_shape=target_shape, stride=16)
        print("Upsampling shape", conv_up.get_shape())
        annotation_pred = tf.argmax(conv_up, dimension=3, name="prediction")
        probabilities = tf.nn.softmax(conv_up)
    return tf.expand_dims(annotation_pred, dim=3), conv_up, probabilities


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    if FLAGS.mode == "test":
        image = tf.placeholder(
            tf.float32, shape=[None,IMAGE_SIZE , IMAGE_SIZE, 4], name="input_image")
    else:
        image = tf.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 4], name="input_image")
    annotation = tf.placeholder(
        tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation, logits, final_probabilities = inference(
        image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(
        annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(
        pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(
                                                                              annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    correct_pred = tf.equal(
        tf.cast(pred_annotation, tf.uint8), tf.cast(annotation, tf.uint8))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar("entropy", loss)
    tf.summary.scalar("accuracy", accuracy)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': False, 'resize_size': IMAGE_SIZE}

    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(
            train_records, image_options)
        validation_dataset_reader = dataset.BatchDatset(
            valid_records, image_options)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=True, device_count={'GPU': 1}))

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":

        train_row = 0
        valid_row = 0
        train_record = pd.DataFrame(columns=['itr', 'train_loss', 'train_acc'])
        valid_record = pd.DataFrame(columns=['itr', 'valid_loss', 'valid_acc'])

        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(
                FLAGS.batch_size)
            feed_dict = {image: train_images,
                         annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            #test = sess.run(pred_annotation, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_acc = sess.run(accuracy, feed_dict=feed_dict)
                train_loss, summary_str = sess.run(
                    [loss, summary_op], feed_dict=feed_dict)
                print("Step: %-6d, Train_loss: %-10g, Train_Acc: %-10g" %
                      (itr, train_loss, train_acc))
                summary_writer.add_summary(summary_str, itr)

                train_record.loc[train_row] = [itr, train_loss, train_acc]
                train_row += 1
            if itr % 10 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(
                    FLAGS.batch_size)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                valid_acc = sess.run(accuracy, feed_dict={image: valid_images, annotation: valid_annotations,
                                                          keep_probability: 1.0})
                print("Step: %-6d, Valid_loss: %-10g, Valid_acc: %-10g <--- %s" %
                      (itr, valid_loss, valid_acc, datetime.datetime.now()))

                if valid_acc > 0.90 or itr % 500 == 0:
                    saver.save(sess, "D:/DeepSEG/model.ckpt", itr)

                valid_record.loc[valid_row] = [itr, valid_loss, valid_acc]
                valid_row += 1

            train_record.to_csv(
                "D:/DeepSEG/train_record.csv", index=False, sep=',', mode='w')
            valid_record.to_csv(
                "D:/DeepSEG/valid_record.csv", index=False, sep=',', mode='w')

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(
            FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(
                np.uint8), FLAGS.logs_dir, name="inp_" + str(5 + itr))
            utils.save_image(valid_annotations[itr].astype(
                np.uint8), FLAGS.logs_dir, name="gt_" + str(5 + itr))
            utils.save_image(pred[itr].astype(np.uint8),
                             FLAGS.logs_dir, name="pred_" + str(5 + itr))
            print("Saved image: %d" % itr)

    # 更新：测试
    elif FLAGS.mode == "test":
        test_dir = 'D:/DeepSEG/test1/*.npy'
        test_file_list = glob.glob(test_dir)
        for f in test_file_list:
            test_path = f
            image_test = np.load(test_path)
            test_image = image_test.reshape(
                (1, image_test.shape[0], image_test.shape[1], image_test.shape[2]))

            start_time = time.time()

            last_probabilities = sess.run(final_probabilities, feed_dict={
                                          image: test_image, keep_probability: 1.0})
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(last_probabilities.squeeze()[:, :, 2], cmap=cm.jet)
            plt.show()
            processed_probabilities = last_probabilities.squeeze()
            processed_probabilities = processed_probabilities.transpose(
                (2, 0, 1))
            unary = softmax_to_unary(processed_probabilities)
            unary = np.ascontiguousarray(unary)
            d = dcrf.DenseCRF(image_test.shape[0] * image_test.shape[1], 6)
            d.setUnaryEnergy(unary)

            feats = create_pairwise_gaussian(
                sdims=(5, 5), shape=image_test.shape[:2])  # 10
            d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)  # 3

            feats = create_pairwise_bilateral(sdims=(25, 25), schan=(
                10, 10, 10, 50), img=image_test, chdim=2)  # 50 20
            d.addPairwiseEnergy(feats, compat=5, kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)  # 10

            Q = d.inference(10)
            pred_image = np.argmax(Q, axis=0)
            end_time = time.time()
            test_time = end_time - start_time

            image_pred = pred_image.reshape(
                (image_test.shape[0], image_test.shape[1]))

            rgb_img = np.zeros(
                [image_pred.shape[0], image_pred.shape[1], 3], dtype=np.uint8)

            for i in np.arange(0, image_pred.shape[0]):
                for j in np.arange(0, image_pred.shape[1]):
                    if image_pred[i][j] == 0:
                        rgb_img[i][j][0] = 255
                        rgb_img[i][j][1] = 255
                        rgb_img[i][j][2] = 255
                    elif image_pred[i][j] == 1:
                        rgb_img[i][j][0] = 255
                        rgb_img[i][j][1] = 0
                        rgb_img[i][j][2] = 0
                    elif image_pred[i][j] == 2:
                        rgb_img[i][j][0] = 0
                        rgb_img[i][j][1] = 0
                        rgb_img[i][j][2] = 255
                    elif image_pred[i][j] == 3:
                        rgb_img[i][j][0] = 0
                        rgb_img[i][j][1] = 255
                        rgb_img[i][j][2] = 0
                    elif image_pred[i][j] == 4:
                        rgb_img[i][j][0] = 255
                        rgb_img[i][j][1] = 255
                        rgb_img[i][j][2] = 0
                    elif image_pred[i][j] == 5:
                        rgb_img[i][j][0] = 0
                        rgb_img[i][j][1] = 255
                        rgb_img[i][j][2] = 255
            save_path = f[:-4] + '_test.tif'
            misc.imsave(save_path, rgb_img)
            print('Test image ' + f[-27:-4] + ' saved...')
            print('Time: %f' % test_time)


if __name__ == "__main__":
    tf.app.run()
