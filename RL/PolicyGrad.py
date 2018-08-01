import numpy as np
import tensorflow as tf

class policy_grad:
    def __init__(self,sdim,adim,lr,dr):

        self.sdim = sdim
        self.adim = adim
        self.lr = lr
        self.dr = dr

        self.o_tr=[]
        self.a_tr=[]
        self.r_tr =[]

        self.policy_network()


        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

    def policy_network(self):
        # Create placeholders
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, [self.sdim, None], name="X")
            self.Y = tf.placeholder(tf.float32, [self.adim, None], name="Y")
            self.dis_r_tr = tf.placeholder(tf.float32, shape=None, name="actions_value")

        # Initialize parameters
        in_layer = 10
        hid_layer1 = 8
        hl2 = 7
        out_layer = self.adim
        with tf.name_scope('parameters'):
            W1 = tf.get_variable("W1", [in_layer, self.sdim], initializer = tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable("b1", [in_layer, 1], initializer = tf.contrib.layers.xavier_initializer())
            W2 = tf.get_variable("W2", [hid_layer1, in_layer], initializer = tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable("b2", [hid_layer1, 1], initializer = tf.contrib.layers.xavier_initializer())
            W3 = tf.get_variable("W3", [hl2, hid_layer1], initializer = tf.contrib.layers.xavier_initializer())
            b3 = tf.get_variable("b3", [hl2, 1], initializer = tf.contrib.layers.xavier_initializer())
            W4 = tf.get_variable("W4", [out_layer, hl2], initializer = tf.contrib.layers.xavier_initializer())
            b4 = tf.get_variable("b4", [out_layer, 1], initializer = tf.contrib.layers.xavier_initializer())


        Z1 = tf.add(tf.matmul(W1,self.X), b1)
        A1 = tf.nn.relu(Z1)
      
        Z2 = tf.add(tf.matmul(W2, A1), b2)
        A2 = tf.nn.relu(Z2)
       
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)       
        
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        #A3 = tf.nn.softmax(Z3)

        # Softmax outputs, we need to transpose as tensorflow nn functions expects them in this shape
        
        logits = tf.transpose(Z4)
        self.outputs = tf.nn.softmax(logits, name='A4')

        labels = tf.transpose(self.Y)
        log_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        # log_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)

        loss = tf.reduce_mean(log_loss * self.dis_r_tr)  # reward guided loss

        self.train_nn = tf.train.AdamOptimizer(self.lr).minimize(loss)




    def take_act(self, obs):

        obs = obs[:,np.newaxis]
        act_prob = self.sess.run(self.outputs, feed_dict={self.X: obs})
        act = np.random.choice(range(len(act_prob.ravel())), p=act_prob.ravel())
        #act = np.random.choice(act_prob.ravel(), p=act_prob.ravel()) # box action
        return act

    def build_trajectory(self, s, a, r):

        self.o_tr.append(s)

        act = np.zeros(self.adim)
        act[a] = 1
        self.a_tr.append(act)
        self.r_tr.append(r)

    def dis_r(self):
        discount_r_tr = np.zeros_like(self.r_tr)
        
        discount_r_tr[-1]= self.r_tr[-1]
        for t in reversed(range(len(self.r_tr)-1)):
            discount_r_tr[t]=discount_r_tr[t+1]*self.dr+self.r_tr[t] # Q(t)= r(st,at)+gamma*Q(t+1)
        
        discount_r_tr-=np.mean(discount_r_tr)
        discount_r_tr/=np.std(discount_r_tr)
        return discount_r_tr

    def learn(self):
        dis_r_tr = self.dis_r()
        self.sess.run(self.train_nn, feed_dict={
             self.X: np.vstack(self.o_tr).T,
             self.Y: np.vstack(np.array(self.a_tr)).T,
             self.dis_r_tr: dis_r_tr
        })

        self.o_tr=[]
        self.a_tr=[]
        self.r_tr =[]


        return dis_r_tr
