import tensorflow as tf

with tf.Session() as sess:

    x = tf.placeholder(tf.float32, shape=[None, 748], name="x")
    y = tf.placeholder(tf.float32, [None, 10], name="y")
    w = tf.placeholder(tf.float32, [748, 10], name="weights_in")

    W = tf.Variable(tf.zeros([748, 10]), name="weights")
    b = tf.Variable(tf.zeros([10]))

    y_out = tf.add(tf.matmul(x, W), b, name="y_out")

    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_out), reduction_indices=[1]))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_out))
    train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy, name="train")

    correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="test")

    init = tf.variables_initializer(tf.global_variables(), name="init")

    tf.train.write_graph(sess.graph_def,
                         './',
                         'mnist_mlp.pb', as_text=False)


"""
Log.d(" ", " ");
Log.d("Weights Equal", "" + w.equals(w_new));
Log.d("Original Weights", ""+w[0]+" "+w[1]+" "+w[2]);
Log.d("New Weights", ""+w[0]+" "+w[1]+" "+w[2]);
Log.d(" ", "");
w = w_new;
"""