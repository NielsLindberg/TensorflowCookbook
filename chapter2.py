import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import DataSources as ds

def operations_in_computational_graph ():

    sess = tf.Session()
    x_vals = np.array([1., 3., 5., 7., 9.])
    x_data = tf.placeholder(tf.float32)
    m_const = tf.constant(3.)
    my_product = tf.multiply(x_data, m_const)
    for x_val in x_vals:
        print(sess.run(my_product, feed_dict={x_data: x_val}))

def layering_nested_operations ():

    sess = tf.Session()
    my_array = np.array([[1.,3.,5.,7.,9.],[-2.,0.,2.,4.,6.],[-6.,-3.,0.,3.,6.]])
    x_vals = np.array([my_array, my_array + 1])
    x_data = tf.placeholder(tf.float32, shape=(3,5))

    m1 = tf.constant([[1.], [0.], [-1.], [2.], [4.]])
    m2 = tf.constant([[2.0]])
    a1 = tf.constant([[10.]])

    prod1 = tf.matmul(x_data, m1)
    prod2 = tf.matmul(prod1, m2)
    add1 = tf.add(prod2, a1)

    for x_val in x_vals:
        print(sess.run(add1, feed_dict={x_data: x_val}))

def custom_layer(input_matrix):
    input_matrix_squeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1.,2.],[-1.,3.]])
    b = tf.constant(1., shape=[2,2])
    temp1 = tf.matmul(A, input_matrix_squeezed)
    temp = tf.add(temp1, b)
    return tf.sigmoid(temp)

def working_with_multiple_layers ():
    sess = tf.Session()
    x_shape = [1, 4, 4, 1]
    x_val = np.random.uniform(size=x_shape)
    x_data = tf.placeholder(tf.float32, shape=x_shape)
    my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
    my_strides = [1, 2, 2, 1]
    mov_avg_layer = tf.nn.conv2d(x_data, my_filter, my_strides,
                                 padding='SAME', name='Moving_Avg_Window')
    with tf.name_scope('Custom_Layer') as scope:
        custom_layer1 = custom_layer(mov_avg_layer)
        print(sess.run(custom_layer1, feed_dict={x_data: x_val}))

def implementing_loss_functions ():
    sess = tf.Session()
    x_vals = tf.linspace(-1.,1.,500)
    target = tf.constant(0.)

    ##L2 Loss
    l2_y_vals = tf.square(target - x_vals)
    l2_y_out = sess.run(l2_y_vals)

    ##L1 Loss
    l1_y_vals = tf.abs(target - x_vals)
    l1_y_out = sess.run(l1_y_vals)


    ##Pseudo-Huber
    delta1 = tf.constant(0.25)
    delta2 = tf.constant(5.)

    phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals)/delta1)) - 1.)
    phuber1_y_out = sess.run(phuber1_y_vals)

    phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((target - x_vals) / delta2)) - 1.)
    phuber2_y_out = sess.run(phuber2_y_vals)

    ##New values and target
    x_vals = tf.linspace(-3., 5., 500)
    target = tf.constant(1.)
    targets = tf.fill([500,], 1.)

    ##Hinge  Loss
    hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target, x_vals))
    hinge_y_out = sess.run(hinge_y_vals)

    ##Cross Entropy
    xentropy_y_vals = - tf.multiply(target, tf.log(x_vals) - tf.multiply((1. - target), tf.log(1. - x_vals)))
    xentropy_y_out = sess.run(xentropy_y_vals)

    ##Sigmoid cross entropy
    xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_vals, labels=targets)
    xentropy_sigmoid_y_out = sess.run(xentropy_sigmoid_y_vals)

    ##Weighted cross entropy loss
    weight = tf.constant(0.5)
    xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(x_vals, targets, weight)
    xentropy_weighted_y_out = sess.run(xentropy_weighted_y_vals)

    ##softmax cross-entropy
    unscaled_logits = tf.constant([[1.,-3.,10.]])
    target_dist = tf.constant([[0.1,0.02,0.88]])
    softmax_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=unscaled_logits, labels=target_dist)

    ##Sparse softmax cross entropy
    unscaled_logits = tf.constant([[1.,-3.,10.]])
    sparse_target_dist = tf.constant([2])
    sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=unscaled_logits, labels=sparse_target_dist)

    x_array = sess.run(x_vals)
    plt.plot(x_array, l2_y_out, 'b-', label='L2 Loss')
    plt.plot(x_array, l1_y_out, 'r--', label='L1 Loss')
    plt.plot(x_array, phuber1_y_out, 'k-', label='P-Huber Loss (0.25)')
    plt.plot(x_array, phuber2_y_out, 'g-', label='P-Huber Loss (0.5)')
    plt.ylim(-0.2,0.4)
    plt.legend(loc='lower right', prop={'size': 11})
    plt.show()

    x_array = sess.run(x_vals)
    plt.plot(x_array, hinge_y_out, 'b-', label='Hinge Loss')
    plt.plot(x_array, xentropy_y_out, 'r--', label='Cross Entropy Loss')
    plt.plot(x_array, xentropy_sigmoid_y_out, 'k-', label='Cross Entropy Sigmoid Loss')
    plt.plot(x_array, xentropy_weighted_y_out, 'g:', label='Weighted Cross Entropy Loss(x0.5)')
    plt.ylim(-1.5, 3)
    plt.legend(loc='lower right', prop={'size': 11})
    plt.show()

def implementing_back_propagation ():
    sess = tf.Session()
    x_vals = np.random.normal(1, 0.1, 100)
    y_vals = np.repeat(10., 100)
    x_data = tf.placeholder(shape=[1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[1], dtype=tf.float32)
    A = tf.Variable(tf.random_normal(shape=[1]))

    my_output = tf.multiply(x_data, A)

    loss = tf.square(my_output - y_target)

    init = tf.initialize_all_variables()
    sess.run(init)

    my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
    train_step = my_opt.minimize(loss)

    for i in range(100):
        rand_index = np.random.choice(100)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        if( i + 1)%25==0:
            print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
            print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))

    ## Simple Classification
    ops.reset_default_graph()
    sess = tf.Session()

    x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3,1, 50)))
    y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1.,50)))
    x_data = tf.placeholder(shape=[1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[1], dtype=tf.float32)
    A = tf.Variable(tf.random_normal(mean=10, shape=[1]))
    my_output = tf.add(x_data, A)

    my_output_expanded = tf.expand_dims(my_output, 0)
    y_target_expanded = tf.expand_dims(y_target, 0)

    init = tf.initialize_all_variables()
    sess.run(init)

    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output_expanded, labels=y_target_expanded)
    my_opt = tf.train.GradientDescentOptimizer(0.05)

    train_step = my_opt.minimize(xentropy)

    for i in range(1400):
        rand_index = np.random.choice(100)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]

        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        if (i+i)%200==0:
            print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
            print('Loss = ' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))

def working_with_batch_and_stochastic_training ():
    sess = tf.Session()
    batch_size = 20
    x_vals = np.random.normal(1, 0.1, 100)
    y_vals = np.repeat(10., 100)
    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    A = tf.Variable(tf.random_normal(shape=[1,1]))

    my_output = tf.matmul(x_data, A)

    init = tf.global_variables_initializer()
    sess.run(init)

    loss = tf.reduce_mean(tf.square(my_output - y_target))
    my_opt = tf.train.GradientDescentOptimizer(0.2)
    train_step = my_opt.minimize(loss)

    loss_batch = []
    for i in range(100):
        rand_index = np.random.choice(100, size=batch_size)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        if(i+1)%5==0:
            print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
            temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
            print('Loss = ' + str(temp_loss))
            loss_batch.append(temp_loss)

    loss_stochastic = []
    for i in range(100):
        rand_index = np.random.choice(100, size=1)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        if(i+1)%5==0:
            print('Step #' +str(i+1) + ' A = ' + str(sess.run(A)))
            temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
            print('Loss = ' + str(temp_loss))
            loss_stochastic.append(temp_loss)

    plt.plot(range(0,100,5), loss_stochastic, 'b-', label='Stochastic Loss')
    plt.plot(range(0,100,5), loss_batch, 'r--', label='Batch Loss')
    plt.legend(loc='upper right', prop={'size': 11})
    plt.show()

def combining_everything_together():
    iris = ds.iris_data()
    sess = tf.Session()
    binary_target = np.array([1. if x==0 else 0. for x in iris.target])
    iris_2d = np.array([[x[2], x[3]] for x in iris.data])
    batch_size = 20

    x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    A = tf.Variable(tf.random_normal(shape=[1, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    my_mult = tf.matmul(x2_data, A)
    my_add = tf.add(my_mult, b)
    my_output = tf.subtract(x1_data, my_add)

    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output, labels=y_target)
    my_opt = tf.train.GradientDescentOptimizer(0.05)
    train_step = my_opt.minimize(xentropy)
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
        rand_index = np.random.choice(len(iris_2d), size=batch_size)
        rand_x = iris_2d[rand_index]
        rand_x1 = np.array([[x[0]] for x in rand_x])
        rand_x2 = np.array([[x[1]] for x in rand_x])
        rand_y = np.array([[y] for y in binary_target[rand_index]])
        sess.run(train_step, feed_dict={x1_data: rand_x1, x2_data: rand_x2, y_target: rand_y})
        if (i+1)%200==0:
            print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) +
                ', b = ' + str(sess.run(b)))

    [[slope]] = sess.run(A)
    [[intercept]] = sess.run(b)
    x = np.linspace(0, 3, num=50)
    ablineValues = []
    for i in x:
        ablineValues.append(slope*i+intercept)
    setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==1]
    setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==1]
    non_setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==0]
    non_setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==0]
    plt.plot(setosa_x, setosa_y, 'rx', ms=10, mew=2, label='setosa')
    plt.plot(non_setosa_x, non_setosa_y, 'ro', label='Non-Setosa')
    plt.plot(x, ablineValues, 'b-')
    plt.xlim([0.0, 2.7])
    plt.ylim([0.0, 7.1])
    plt.suptitle('Linear Separator For I.setosa', fontsize=20)#
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend(loc='lower right')
    plt.show()

def evaluating_models():
    sess = tf.Session()
    x_vals = np.random.normal(1, 0.1, 100)
    y_vals = np.repeat(10., 100)
    x_data = tf.placeholder(shape=[None,1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    batch_size = 25
    train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
    test_indices = np.array(list(set(range(len(x_vals)))-set(train_indices)))
    x_vals_train = x_vals[train_indices]
    x_vals_test = x_vals[test_indices]
    y_vals_train = y_vals[train_indices]
    y_vals_test = y_vals[test_indices]
    A = tf.Variable(tf.random_normal(shape=[1,1]))
    my_output = tf.matmul(x_data, A)
    loss = tf.reduce_mean(tf.square(my_output - y_target))
    init = tf.global_variables_initializer()
    sess.run(init)
    my_opt = tf.train.GradientDescentOptimizer(0.02)
    train_step = my_opt.minimize(loss)

    for i in range(100):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = np.transpose([x_vals_train[rand_index]])
        rand_y = np.transpose([y_vals_train[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        if (i+1)%25==0:
            print('Step #' +str(i+1) + ' A = ' + str(sess.run(A)))
            print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))

    mse_test = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_test]), y_target: np.transpose([y_vals_test])})
    mse_train = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_train]), y_target: np.transpose([y_vals_train])})
    print('MSE on test: ' + str(np.round(mse_test,2)))
    print('MSE on train: ' + str(np.round(mse_train, 2)))

    ops.reset_default_graph()
    sess = tf.Session()
    batch_size = 25
    x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(2, 1, 50)))
    y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
    x_data = tf.placeholder(shape=[1, None], dtype=tf.float32)
    y_target = tf.placeholder(shape=[1, None], dtype=tf.float32)
    train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
    test_indices = np.array(list(set(range(len(x_vals)))-set(train_indices)))
    x_vals_train = x_vals[train_indices]
    x_vals_test = x_vals[test_indices]
    y_vals_train = y_vals[train_indices]
    y_vals_test = y_vals[test_indices]
    A = tf.Variable(tf.random_normal(mean=10, shape=[1]))
    my_output = tf.add(x_data, A)
    init = tf.global_variables_initializer()
    sess.run(init)
    xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output, labels=y_target))
    my_opt = tf.train.GradientDescentOptimizer(0.05)
    train_step = my_opt.minimize(xentropy)

    for i in range(1800):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = [x_vals_train[rand_index]]
        rand_y = [y_vals_train[rand_index]]
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        if(i+1)%200==0:
            print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)))
            print('Loss = ' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))

    y_prediction = tf.squeeze(tf.round(tf.nn.sigmoid(tf.add(x_data, A))))

    correct_prediction = tf.equal(y_prediction, y_target)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc_value_test = sess.run(accuracy, feed_dict={x_data: [x_vals_test], y_target: [y_vals_test]})
    acc_value_train = sess.run(accuracy, feed_dict={x_data: [x_vals_train], y_target: [y_vals_train]})
    print('Accuracy on train set: ' + str(acc_value_train))
    print('Acurracy on test set: ' + str(acc_value_test))

    A_result = sess.run(A)
    bins = np.linspace(-5, 5, 50)
    plt.hist(x_vals[0:50], bins, alpha=0.5, label='N(-1,1)', color='white', edgecolor='black')
    plt.hist(x_vals[50:100], bins[0:50], alpha=0.5, label='N(2,1)', color='red', edgecolor='black')
    plt.plot((A_result, A_result), (0, 8), 'k--', linewidth=3, label='A = '+str(np.round(A_result, 2)))
    plt.legend(loc='upper right')
    plt.title('Binary Classifier Accuracy=' + str(np.round(acc_value_train, 2)))
    plt.show()






