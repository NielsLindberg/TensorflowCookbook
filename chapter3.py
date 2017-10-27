import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import DataSources as ds

def using_the_matrix_inverse_method():
    sess = tf.Session()
    x_vals = np.linspace(0, 10, 100)
    y_vals = x_vals + np.random.normal(0,1,100)
    x_vals_column = np.transpose(np.matrix(x_vals))
    ones_column = np.transpose(np.matrix(np.repeat(1,100)))
    A = np.column_stack((x_vals_column, ones_column))
    b = np.transpose(np.matrix(y_vals))

    A_tensor = tf.constant(A)
    b_tensor = tf.constant(b)

    tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
    tA_A_inv = tf.matrix_inverse(tA_A)

    product = tf.matmul(tA_A_inv, tf.transpose(A_tensor))

    solution = tf.matmul(product, b_tensor)
    solution_eval = sess.run(solution)

    slope = solution_eval[0][0]
    y_intercept = solution_eval[1][0]
    print('slope: ' + str(slope))
    print('y_intercept : ' + str(y_intercept))

    best_fit = []
    for i in x_vals:
        best_fit.append(slope*i+y_intercept)
    plt.plot(x_vals, y_vals, 'o', label='Data')
    plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
    plt.legend(loc='upper left')
    plt.show()

def implementing_a_decomposition_method():
    sess = tf.Session()
    x_vals = np.linspace(0,10,100)
    y_vals = x_vals + np.random.normal(0, 1, 100)
    x_vals_column = np.transpose(np.matrix(x_vals))
    ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
    A = np.column_stack((x_vals_column, ones_column))
    b = np.transpose(np.matrix(y_vals))
    A_tensor = tf.constant(A)
    b_tensor = tf.constant(b)
    tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
    L = tf.cholesky(tA_A)
    tA_b = tf.matmul(tf.transpose(A_tensor), b_tensor)
    sol1 = tf.matrix_solve(L, tA_b)
    sol2 = tf.matrix_solve(tf.transpose(L), sol1)

    solution_eval = sess.run(sol2)
    slope = solution_eval[0][0]
    y_intercept = solution_eval[1][0]
    print('slope: ' + str(slope))
    print('y_intercept: ' + str(y_intercept))

    best_fit = []
    for i in x_vals:
        best_fit.append(slope*i+y_intercept)
    plt.plot(x_vals, y_vals, 'o', label='Data')
    plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
    plt.legend(loc='upper left')
    plt.show()

def learning_the_tensorflow_way_of_linear_regression():
    #Load data
    iris = ds.iris_data()
    sess = tf.Session()
    #Create numpy array (input) of index 3 value of data
    x_vals = np.array([x[3] for x in iris.data])
    #Create numpy array (target) of index 0 value of data
    y_vals = np.array([y[0] for y in iris.data])
    #hyperparameter determining how much to adjust weights in backpropegation
    learning_rate = 0.05
    #How many inputs to run through network at each training step
    batch_size = 25
    #placeholder for input data (batches are parsed to this on each training step)
    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    #placeholder for target data (batches are parsed to this on each training step)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    #variable for weights in model adjusted through back propegation
    A = tf.Variable(tf.random_normal(shape=[1,1]))
    #variable for bias in model, adjusted through back propegation
    b = tf.Variable(tf.random_normal(shape=[1,1]))
    #model (how placeholders and variables relate to match target)
    model_output = tf.add(tf.matmul(x_data, A), b)
    #how we express the difference between target and model
    #in this case since the target is a continuous value, reduce mean is used.
    loss = tf.reduce_mean(tf.square(y_target - model_output))
    init = tf.global_variables_initializer()
    sess.run(init)
    my_opt = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = my_opt.minimize(loss)

    loss_vec = []
    for i in range(100):
        rand_index = np.random.choice(len(x_vals), size=batch_size)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)
        if(i+1)%25==0:
            print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
            print('Loss = ' + str(temp_loss))
    [slope] = sess.run(A)
    [y_intercept] = sess.run(b)
    best_fit = []
    for i in x_vals:
        best_fit.append(slope*i+y_intercept)

    plt.plot(x_vals, y_vals, 'o', label='Data Points')
    plt.plot(x_vals, best_fit, 'r', label='Best fit line', linewidth=3)
    plt.legend(loc='upper left')
    plt.title('Sepal Length vs Pedal Width')
    plt.xlabel('Pedal Width')
    plt.ylabel('Sepal Length')
    plt.show()
    plt.plot(loss_vec, 'k')
    plt.title('L2 Loss Per Generation')
    plt.xlabel('Generation')
    plt.ylabel('L2 Loss')
    plt.show()

def understanding_loss_functions_in_linear_regression():
    sess = tf.Session()
    iris = ds.iris_data()
    x_vals = np.array([x[3] for x in iris.data])
    y_vals = np.array([y[0] for y in iris.data])
    batch_size = 25
    learning_rate = 0.1
    iterations = 50
    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    A = tf.Variable(tf.random_normal(shape=[1, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))
    model_output = tf.add(tf.matmul(x_data, A), b)
    loss_l1 = tf.reduce_mean(tf.abs(y_target - model_output))
    init = tf.global_variables_initializer()
    sess.run(init)
    my_opt_l1 = tf.train.GradientDescentOptimizer(learning_rate)
    train_step_l1 = my_opt_l1.minimize(loss_l1)
    loss_vec_l1 = []
    for i in range(iterations):
        rand_index = np.random.choice(len(x_vals), size=batch_size)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])
        sess.run(train_step_l1, feed_dict={x_data: rand_x, y_target: rand_y})
        temp_loss_l1 = sess.run(loss_l1, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec_l1.append(temp_loss_l1)
        if (i+1)%25==0:
            print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))

    plt.plot(loss_vec_l1, 'k-', label='L1 Loss')
    plt.title('L1 Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('L1 Loss')
    plt.legend(loc='upper right')
    plt.show()

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)

def implementing_logistic_regression():
    #doesnt work due to access denied in dataset
    birth_data = ds.birth_weight_data()
    sess = tf.Session()

    y_vals = np.array([x[1] for x in birth_data])
    x_vals = np.array([x[2:9] for x in birth_data])
    train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    x_vals_train = x_vals[train_indices]
    x_vals_test = x_vals[test_indices]
    y_vals_train = y_vals[train_indices]
    y_vals_test = y_vals[test_indices]
    x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
    x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))
    batch_size = 25
    x_data = tf.placeholder(shape=[None, 7], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    A = tf.Variable(tf.random_normal(shape=[7,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))
    model_output = tf.add(tf.matmul(x_data, A), b)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
    init = tf.global_variables_initializer()
    sess.run(init)
    my_opt = tf.train.GradientDescentOptimizer(0.01)
    train_step = my_opt.minimize(loss)

    prediction = tf.round(tf.sigmoid(model_output))
    predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
    accuracy = tf.reduce_mean(predictions_correct)

    loss_vec = []
    train_acc = []
    test_acc = []

    for i in range(1500):
        rand_index = np.random.choice(len(x_vals_train), size=batch_size)
        rand_x = x_vals_train[rand_index]
        rand_y = np.transpose([y_vals_train[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)
        temp_acc_train = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
        train_acc.append(temp_acc_train)
        temp_acc_test = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
        test_acc.append(temp_acc_test)

    plt.plot(loss_vec, 'k-')
    plt.title('Cross Entropy Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Cross Entropy Loss')
    plt.show()
    plt.plot(train_acc, 'k-', label='Train Set Accuracy')
    plt.plot(test_acc, 'r--', label='Test Set Accuracy')
    plt.title('train and Test Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()





