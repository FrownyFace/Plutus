import tensorflow as tf
import numpy as np

from utils import load_data

class GAN:
    def __init__(self, params):
        x, y = load_data(params['train_path'])
        assert len(x) == len(y)
        self.batch_size = params['batch_size']
        self.x = [x[i:i+self.batch_size] for i in range(0, len(x), self.batch_size)][::-1]
        self.y = [y[i:i+self.batch_size] for i in range(0, len(y), self.batch_size)][::-1]
        self.epochs = params['epochs']
        self.lr = params['lr']
        self.gen_layers = params['gen_layers']
        self.dis_layers = params['dis_layers']


    def build_model(self, _X, _Y):
        gen_W = [tf.Variable(tf.random_normal([i, j], stddev=0.2), name=k) for i, j, k in self.gen_layers]
        gen_B = [tf.Variable(tf.random_normal([j], stddev=0.2), name=k) for i, j, k in self.gen_layers]

        gen_l = _X
        for W, B in zip(gen_W, gen_B):
            l_out = tf.nn.xw_plus_b(gen_l, W, B)
            gen_l = tf.nn.relu(l_out)

       #p = tf.reshape(gen_l, shape=[])
        loss = tf.reduce_mean(tf.losses.absolute_difference(labels=_Y, predictions=gen_l))
        opt = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return opt, loss, gen_l

    def train(self):
        _X = tf.placeholder(shape=[None, 9], dtype=tf.float32, name="X")
        _Y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="Y")

        '''x_batch, y_batch = tf.train.shuffle_batch(
    [self.x, self.y],
    batch_size=self.batch_size,
    num_threads=1,
    capacity=50000,
    min_after_dequeue=10000
)'''

        opt, loss, p = self.build_model(_X, _Y)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for e in range(self.epochs + 1):
            for x_batch, y_batch in zip(self.x, self.y):
                _, cur_loss, pred = sess.run([opt, loss, p], feed_dict={
                    _X: x_batch,
                    _Y: np.expand_dims(y_batch, axis=1)
                })
            if e % 20 == 0:
                print(e, cur_loss)
        print(y_batch, pred)


if __name__ == '__main__':

    params = {
        'gen_layers': [[9, 50, 'gen_W1'],
                       [50, 50, 'gen_W2'],
                       [50, 1, 'gen_W3']],

        'dis_layers': [[9, 50, 'dis_W1'],
                       [50, 1, 'dis_W2']],

        'lr': 0.001,
        'epochs': 100,
        'train_path': './data/UnderlyingOptionsTrades_2016-06-01.csv',
        'test_path': 'test_o.csv',
        'batch_size': 25,
        'export_base_path': './export_models',

    }
    d = GAN(params)
    d.train()
