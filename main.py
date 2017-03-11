import tensorflow as tf
import numpy as np
import resnet_model
import argparse
import tlfiles

parser = argparse.ArgumentParser(description='Define parameters.')

parser.add_argument('--n_epoch', type=int, default=10)
parser.add_argument('--n_batch', type=int, default=64)
parser.add_argument('--n_img_row', type=int, default=32)
parser.add_argument('--n_img_col', type=int, default=32)
parser.add_argument('--n_img_channels', type=int, default=3)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--n_resid_units', type=int, default=5)
parser.add_argument('--lr_schedule', type=int, default=60)
parser.add_argument('--lr_factor', type=float, default=0.1)

args = parser.parse_args()


def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    """Generate a generator that input a group of example in numpy.array and
    their labels, return the examples and labels by the given batchsize.
    Parameters
    ----------
    inputs : numpy.array
        (X) The input features, every row is a example.
    targets : numpy.array
        (y) The labels of inputs, every row is a example.
    batch_size : int
        The batch size.
    shuffle : boolean
        Indicating whether to use a shuffling queue, shuffle the dataset before return.
    Examples
    --------
    >>> X = np.asarray([['a','a'], ['b','b'], ['c','c'], ['d','d'], ['e','e'], ['f','f']])
    >>> y = np.asarray([0,1,2,3,4,5])
    >>> for batch in tl.iterate.minibatches(inputs=X, targets=y, batch_size=2, shuffle=False):
    >>>     print(batch)
    ... (array([['a', 'a'],
    ...        ['b', 'b']],
    ...         dtype='<U1'), array([0, 1]))
    ... (array([['c', 'c'],
    ...        ['d', 'd']],
    ...         dtype='<U1'), array([2, 3]))
    ... (array([['e', 'e'],
    ...        ['f', 'f']],
    ...         dtype='<U1'), array([4, 5]))
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]



class CNNEnv:
    def __init__(self):

        # The data, shuffled and split between train and test sets
        # self.x_train, self.y_train, self.x_test, self.y_test = tlfiles.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)
        self.x_train, self.y_train, self.x_test, self.y_test = tlfiles.load_cifar100_dataset(shape=(-1, 32, 32, 3), plotable=False)

        # Reorder dimensions for tensorflow
        self.mean = np.mean(self.x_train, axis=0, keepdims=True)
        self.std = np.std(self.x_train)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        print('x_train shape:', self.x_train.shape)
        print('x_test shape:', self.x_test.shape)
        print('y_train shape:', self.y_train.shape)
        print('y_test shape:', self.y_test.shape)

        # For generator
        self.num_examples = self.x_train.shape[0]
        self.index_in_epoch = 0
        self.epochs_completed = 0

        # Basic info
        self.batch_num = args.n_batch
        self.num_epoch = args.n_epoch
        self.img_row = args.n_img_row
        self.img_col = args.n_img_col
        self.img_channels = args.n_img_channels
        self.nb_classes = args.n_classes
        self.num_iter = self.x_train.shape[0] // self.batch_num  # per epoch

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        self.batch_size = batch_size

        start = self.index_in_epoch
        self.index_in_epoch += self.batch_size

        if self.index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.x_train = self.x_train[perm]
            self.y_train = self.y_train[perm]

            # Start next epoch
            start = 0
            self.index_in_epoch = self.batch_size
            assert self.batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.x_train[start:end], self.y_train[start:end]

    def train(self, hps):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        sess = tf.InteractiveSession(config=config)

        img = tf.placeholder(tf.float32, shape=[self.batch_num, 32, 32, 3])
        labels = tf.placeholder(tf.int32, shape=[self.batch_num, ])

        # 'train' is for training mode vs. 'eval'
        model = resnet_model.ResNet(hps, img, labels, 'train')
        model.build_graph()

        #acc_var = tf.Variable(tf.zeros([1]))
        # Create a saver.
        saver = tf.train.Saver(tf.trainable_variables())

        merged = model.summaries
        train_writer = tf.summary.FileWriter("/tmp/train_log/2", sess.graph)

        sess.run(tf.global_variables_initializer())
        print('Done initializing variables')
        print('Running model...')

        # Set default learning rate for scheduling
        lr = args.lr

        for j in range(self.num_epoch):
            print('Epoch {}'.format(j+1))

            # Decrease learning rate every args.lr_schedule epoch
            # By args.lr_factor
            if (j + 1) % args.lr_schedule == 0:
                lr *= args.lr_factor

            for i in range(self.num_iter):
                batch = self.next_batch(self.batch_num)
                feed_dict = {img: batch[0],
                             labels: batch[1],
                             model.lrn_rate: lr}
                _, l, ac, summary, lr = sess.run([model.train_op, model.cost, model.acc, merged, model.lrn_rate], feed_dict=feed_dict)
                train_writer.add_summary(summary, i)

                #
                if i % 200 == 0:
                    print('step', i+1)
                    print('Training loss', l)
                    print('Training accuracy', ac)
                    print('Learning rate', lr)
                    
                    checkpoint_file='checkpoints/checkpoint_{}.chk'.format(i)
                    
                    # x = tf.Variable([42.0, 42.1, 42.3], name=’x’)
                    # y = tf.Variable([[1.0, 2.0], [3.0, 4.0]], name=’y’)
                    # not_saved = tf.Variable([-1, -2], name=’not_saved’)
                    # session.run(tf.initialize_all_variables())

                    # print(session.run(tf.all_variables()))
                    saver.save(sess, checkpoint_file)


            print('Running evaluation...')

            test_loss, test_acc, n_batch = 0, 0, 0
            for batch in minibatches(inputs=self.x_test,
                targets=self.y_test,
                batch_size=self.batch_num,
                shuffle=False):

                feed_dict_eval = {img: batch[0], labels: batch[1]}

                loss, ac = sess.run([model.cost, model.acc], feed_dict=feed_dict_eval)
                test_loss += loss
                test_acc += ac
                n_batch += 1

            tot_test_loss = test_loss / n_batch
            tot_test_acc = test_acc / n_batch

            print('   Test loss: {}'.format(tot_test_loss))
            print('   Test accuracy: {}'.format(tot_test_acc))

        print('Completed training and evaluation.')

run = CNNEnv()

hps = resnet_model.HParams(batch_size=run.batch_num,
                           num_classes=run.nb_classes,
                           min_lrn_rate=0.0001,
                           lrn_rate=args.lr,
                           num_residual_units=args.n_resid_units,
                           use_bottleneck=False,
                           weight_decay_rate=0.0002,
                           relu_leakiness=0.1,
                           optimizer='mom')

run.train(hps)
