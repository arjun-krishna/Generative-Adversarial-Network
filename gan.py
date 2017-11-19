"""
@auth : Arjun Krishna
@desc : 1D Generative Adverserial Network
"""
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm

seed = 7
np.random.seed(seed)
tf.set_random_seed(seed)

class DataDistribution(object):
    def __init__(self):
        self.mu = 2
        self.sigma = 0.2

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples

class GeneratorDistribution(object):
    def __init__(self, a):
        self.a = a

    def sample(self, N):
        return np.random.uniform(-1,1,N)

def linear(input, output_dim, scope=None, stddev=0.1):
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'W', 
            [input.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input, w) + b


def generator(input):
    h0 = tf.nn.softplus(linear(input, 5, 'h0'))
    h1 = linear(h0, 1, 'h1')
    return h1


def discriminator(input, minibatch_layer=True):
    h0 = tf.nn.relu(linear(input, 10, 'h0'))
    h1 = tf.nn.relu(linear(h0, 10, 'h1'))

    if minibatch_layer:
        h2 = minibatch(h1)
    else:
        h2 = tf.nn.relu(linear(h1, 10, scope='h2'))

    h3 = tf.sigmoid(linear(h2, 1, scope='h3'))
    return h3


def minibatch(input, num_kernels=5, kernel_dim=3):
    x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    delta = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_delta = tf.reduce_sum(tf.abs(delta), 2)
    features = tf.reduce_sum(tf.exp(-abs_delta), 2)
    return tf.concat([input, features], 1)

def log(x):
    # Ensures the term doesn't go to large negative numbers
    return tf.log(tf.maximum(x, 1e-5))

class GAN(object):
    def __init__(self, params):

        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(None, 1))
            self.G = generator(self.z)

        self.x = tf.placeholder(tf.float32, shape=(None, 1))
        with tf.variable_scope('D'):
            self.D_real = discriminator(self.x)

        with tf.variable_scope('D', reuse=True):
            self.D_fake = discriminator(self.G)

        self.loss_discriminator = tf.reduce_mean(-log(self.D_real) - log(1 - self.D_fake))
        self.loss_generator = tf.reduce_mean(-log(self.D_fake))

        self.discriminator_params = [v for v in tf.trainable_variables() if v.name.startswith('D/')]
        self.generator_params = [v for v in tf.trainable_variables() if v.name.startswith('G/')]

        self.train_op_d = tf.train.AdamOptimizer(1e-2).minimize(self.loss_discriminator, var_list=self.discriminator_params)
        self.train_op_g = tf.train.AdamOptimizer(1e-3).minimize(self.loss_generator, var_list=self.generator_params)


def train(model, data, gen, params):
    anim_frames = []

    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        for step in range(params.num_steps + 1):
            
            x = data.sample(params.batch_size)
            z = gen.sample(params.batch_size)
            loss_d, _, = session.run([model.loss_discriminator, model.train_op_d], {
                model.x: np.reshape(x, (params.batch_size, 1)),
                model.z: np.reshape(z, (params.batch_size, 1))
            })

            z = gen.sample(params.batch_size)
            loss_g, _ = session.run([model.loss_generator, model.train_op_g], {
                model.z: np.reshape(z, (params.batch_size, 1))
            })

            if step % params.log_every == 0:
                print('{}: {:.4f}\t{:.4f}'.format(step, loss_d, loss_g))

            if params.anim_path and (step % params.anim_every == 0):
                anim_frames.append(
                    samples(model, session, data, gen, 3)
                )

        if params.anim_path:
            for i in xrange(len(anim_frames)) :
              save_plot(anim_frames[i], 3, i, i*params.anim_every, params.anim_path)
        else:
            s = samples(model, session, data, gen, 3)
            plot_distributions(s, 3)


def samples(model, session, data, gen, sample_range):
    
    xs = np.linspace(-sample_range, sample_range, 100)
    bins = np.linspace(-sample_range, sample_range, 20)

    # decision boundary
    db = session.run(model.D_real, { model.x: np.reshape( xs, (100, 1))})

    # data distribution
    z = gen.sample(1000)

    # generated samples
    g = session.run(model.G, { model.z: np.reshape(z, (1000, 1))})
    
    mean_g = np.mean(g)
    std_g = np.std(g)

    true_dist = norm.pdf(xs, data.mu, data.sigma)
    gen_dist = norm.pdf(xs, mean_g, std_g)

    return db, z, g, true_dist, gen_dist


def plot_distributions(samps, sample_range):
    db, z, g, true_dist, gen_dist = samps
    
    dx = np.linspace(-sample_range, sample_range, len(db))
    px = np.linspace(-sample_range, sample_range, len(z))

    f, ax = plt.subplots(1)
    
    ax.plot(dx, db, color='orange', label='Discriminator')
    ax.set_ylim(0, 3)
    
    # plt.plot(px, pz, color='green', label='Noise')
    plt.hist(z, 20, normed=1, color='green', label='Noise')
    # plt.plot(px, pg, color='red', label='Gen hist')
    plt.hist(g, 20, normed=1, color='red', label='Gen hist')

    plt.plot(dx, true_dist, color='blue', label='Data dist')
    plt.plot(dx, gen_dist, color='purple', label='Gen dist')

    plt.title('Generative Adversarial Network')
    
    plt.legend()
    plt.show()

def save_plot(samps, sample_range, file, title, dir_path) :
    db, z, g, true_dist, gen_dist = samps
    
    dx = np.linspace(-sample_range, sample_range, len(db))
    px = np.linspace(-sample_range, sample_range, len(z))

    f, ax = plt.subplots(1)
    
    ax.plot(dx, db, color='orange', label='Discriminator')
    ax.set_ylim(0, 3)
    
    # plt.plot(px, pz, color='green', label='Noise')
    plt.hist(z, 20, normed=1, color='green', label='Noise')
    # plt.plot(px, pg, color='red', label='Gen hist')
    plt.hist(g, 20, normed=1, color='red', label='Gen hist')

    plt.plot(dx, true_dist, color='blue', label='Data dist')
    plt.plot(dx, gen_dist, color='purple', label='Gen dist')

    plt.title('Generative Adversarial Network [iteration = '+str(title)+']')
    
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if (dir_path[-1] == '/') :
        plt.savefig(dir_path+str(file)+'.png')
    else :
        plt.savefig(dir_path+'/'+str(file)+'.png')


def main(args):
    model = GAN(args)
    train(model, DataDistribution(), GeneratorDistribution(1), args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=5000,
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='the batch size')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--anim-path', type=str, default=None,
                        help='path to the output animation file')
    parser.add_argument('--anim-every', type=int, default=1,
                        help='save every Nth frame for animation')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())