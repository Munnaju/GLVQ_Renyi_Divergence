import matplotlib.pyplot as plt
import numpy as nump
from sklearn.datasets import load_iris as dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

class GLVQ:

    """ 
    
    Generlaized Learning Vector Quantization (GLVQ)

    Attributes:
        class_prototype:
            The number of prototype of each class to be learned

    """

    def __init__(self, class_prototype):
        """ 
        Inits GLVQ with prototype in each class
        
        """
        self.class_prototype = class_prototype

    new_proto_types = nump.array([])
    prototype_data_labels = nump.array([])

    # define prototypes
    def create_data_prototype(self, data_in, data_labels, class_prototype):
        """
        Calculate prototypes with labels. The calculation is based on the mean or at random based on the
        prototype in each class.

        :param data_in: A n x m matrix of datapoints
        :param data_labels: A n-dimensional vector containing the labels for each datapoint
        :param class_prototype: The number of prototype in each class to be learned. If the number of prototypes
        in each class in 1, the prototypes aer assigned in the mean position, otherwise it is assigned at random.

        :return:
            pro_labels:
                A n-dimensional vector having the labels for every prototype
            Prototypes:
                A n x m matrix of prototypes for the training purpose.
        """

        # pro_labels are
        pro_labels = nump.unique(data_labels)
        pro_labels = list(pro_labels) * class_prototype

        # prototypes are
        prototype_data_labels = nump.expand_dims(pro_labels, axis=1)
        expanded_data = nump.expand_dims(nump.equal(prototype_data_labels, data_labels),
                                         axis=2)

        count = nump.count_nonzero(expanded_data, axis=1)
        all_proto = nump.where(expanded_data, data_in, 0)
        prototypes = nump.sum(all_proto, axis=1) / count

        self.prototype_data_labels = pro_labels
        return self.prototype_data_labels, prototypes

    #L1 normalization
    def normalize_data(selfs, data_in):
    
        """
        Normalizing the data so that the data is between the range 0 and 1 and also we can further the data as
        a probability distribution.

        :param data_in: A nx m matrix of our input data.
        :return:
            Data_normalized: A n x m matrix with values between 0 and 1
        """
        Data_normalizer = Normalizer(norm='l1').fit(data_in)
        Data_normalized = Data_normalizer.transform(data_in)
        return Data_normalized


    def Renyi_Divergence(self, data_in, prototypes):
    
        """
        Calculate the Renyi divergence between datapoints and prototypes.
        
        :param data_in: A n x m matrix of datapoints.
        :param prototypes: A n x m matrix of prototyes of each class
        :return: A n x m matrix with Renyi diverence between datapoints and prototypes
        
        """

        expanded_data = nump.expand_dims(data_in, axis=1)
        Renyi_div = nump.sum( nump.where(expanded_data != 0,  nump.square(expanded_data) /prototypes , 0) , axis=2) #KL_Divergence formula
        final_Renyi_div = nump.log(Renyi_div)
        return final_Renyi_div


    # define delta_w_minus
    def change_in_w_minus(self, data_in, learning_rate, classifier,
                          w_minus, w_minus_index, d_plus, d_minus):
                          
                          
        """
        Calculate the update of prototypes

        :param data_in: A n x m matrix of datapoints.
        :param learning_rate: Learning rate (step size)
        :param classifier: Classify the vector as a winner or runner up prototype.
        :param w_minus: A m x n matrix of nearest incorrect matching prototypes.
        :param w_minus_index: A n-dimensional vector having the indices for nearest prototype to datapoints with different label.
        :param d_plus: A n-dimesional vector having the distance between datapoints and prototypes with the same label.
        :param d_minus: A n-dimesional vector having the distance between datapoints and prototypes with different label.
        :return: The result of the updated prototype after calculating the update of prototypes with the same or different label.
        
        """

        lai = (2) * (d_plus / (nump.square(d_plus + d_minus))) * \
              (self.sigmoid_calc(classifier)) * (1 - self.sigmoid_calc(classifier))

        expanded_data = nump.expand_dims(lai, axis=1)
        change_w_minus = (expanded_data) * (data_in - w_minus) * learning_rate

        # index of w_minus
        w_minus_idx = nump.unique(w_minus_index)
        w_minus_idx = nump.expand_dims(w_minus_idx, axis=1)

        row_change_in_w = nump.column_stack((w_minus_index, change_w_minus))
        chk_val = nump.equal(row_change_in_w[:, 0], w_minus_idx)
        chk_val = nump.expand_dims(chk_val, axis=2)
        chk_val = nump.where(chk_val, change_w_minus, 0)
        sum_in_w_minus = nump.sum(chk_val, axis=1)
        return sum_in_w_minus, w_minus_idx

    # define Sigmoid function
    def sigmoid_calc(self, j, beta=10):
        """
        Calculate the sigmoid activation function.
        
        :param j: A n x m matrix of datapoints or any value
        :param beta: The value of the parameter is taken as 10
        :return: It returns values in the range 0 to 1
        
        """
        return (1 / (1 + nump.exp(-beta * j)))

    # define delta_w_plus
    def change_in_w_plus(self, data_in, learning_rate, classifier,
                         w_plus, w_plus_index, d_plus, d_minus):
        """
        Calculate the update of prototypes

        :param data_in: A n x m matrix of datapoints.
        :param learning_rate: Learning rate (step size)
        :param classifier: Classify the vector as a winner or runner up prototype.
        :param w_plus: A m x n matrix of nearest correct matching prototypes.
        :param w_plus_index: A n-dimensional vector having the indices for nearest prototype to datapoints with the same label.
        :param d_plus: A n-dimesional vector having the distance between datapoints and prototypes with the same label.
        :param d_minus: A n-dimesional vector having the distance between datapoints and prototypes with different label.
        :return: The result of the updated prototype after calculating the update of prototypes with the same or different label.
        
        """

        lai = (2) * (d_minus / (nump.square(d_plus + d_minus))) * \
              (self.sigmoid_calc(classifier)) * (1 - self.sigmoid_calc(classifier))

        expanded_data = nump.expand_dims(lai, axis=1)
        change_w_plus = expanded_data * (data_in - w_plus) * learning_rate

        # index of w_plus
        w_plus_idx = nump.unique(w_plus_index)
        w_plus_idx = nump.expand_dims(w_plus_idx, axis=1)

        row_change_in_w = nump.column_stack((w_plus_index, change_w_plus))
        chk_val = nump.equal(row_change_in_w[:, 0], w_plus_idx)
        chk_val = nump.expand_dims(chk_val, axis=2)
        chk_val = nump.where(chk_val, change_w_plus, 0)
        sum_in_w_plus = nump.sum(chk_val, axis=1)
        return sum_in_w_plus, w_plus_idx

    # define calculate_d_minus
    def calculate_d_minus(self, data_labels, prot_labels,
                          pro_types, Renyi_div):
        """
        Calculate the distance between data points and prototypes

        :param data_labels: A n-dimensional vector containing the labels for each datapoint.
        :param prot_labels: A n-dimensional vector containing the labels for each prototype.
        :param pro_types: A n x m matrix of prototyes of each class.
        :param Renyi_div: A n x m matrix with Renyi divergence between datapoints and prototypes.
        :return:
            d_minus: A n-dimensional vector having distance between datapoints and prototypes with different label.
            w_minus: A m x n matrix of nearest incorrect matching prototypes.
            w_minus_index: A n-dimensional vector having the indices for nearest prototype to datapoints with different label.

        """
        expanded_data = nump.expand_dims(prot_labels, axis=1)
        label_transpose = nump.transpose(nump.not_equal(expanded_data,
                                                        data_labels))

        # distance of non matching pro_types
        minus_dist = nump.where(label_transpose, Renyi_div, nump.inf)
        d_minus = nump.min(minus_dist, axis=1)

        # index of minimum distance for non best matching pro_types
        w_minus_index = nump.argmin(minus_dist, axis=1)
        w_minus = pro_types[w_minus_index]
        return d_minus, w_minus, w_minus_index

    # define calculate_d_plus
    def calculate_d_plus(self, data_labels, prot_labels,
                         pro_types, Renyi_div):
        """
        Calculate the distance between data points and prototypes

        :param data_labels: A n-dimensional vector containing the labels for each datapoint.
        :param prot_labels: A n-dimensional vector containing the labels for each prototype.
        :param pro_types: A n x m matrix of prototyes of each class.
        :param Renyi_div: A n x m matrix with Renyi divergence between datapoints and prototypes.
        :return:
            d_plus: A n-dimensional vector having distance between datapoints and prototypes with the same label.
            w_plus: A m x n matrix of nearest correct matching prototypes.
            w_plus_index: A n-dimensional vector having the indices for nearest prototype to datapoints with the same label.
        
        """
        expanded_data = nump.expand_dims(prot_labels, axis=1)
        label_transpose = nump.transpose(nump.equal(expanded_data, data_labels))

        # distance of matching pro_types
        plus_dist = nump.where(label_transpose, Renyi_div, nump.inf)
        d_plus = nump.min(plus_dist, axis=1)

        # index of minimum distance for best matching pro_types
        w_plus_index = nump.argmin(plus_dist, axis=1)
        w_plus = pro_types[w_plus_index]
        return d_plus, w_plus, w_plus_index

    # data predictor function
    def data_predictor(self, input_value):
        """
        
        The prediction of the labels for the data. The data are represented by the test-to-training distance matrix. Every
        datapoint will be assigned to the closest prototype.

        :param input_value: A n x m matrix of distances from the test to the training datapoints.
        :return:
            y_label - A n-dimensional vector having the predicted labels for each and every datapoint.
        
        """
        input_value = self.normalize_data(input_value)
        prototypes = self.new_proto_types
        Renyi_div = self.Renyi_Divergence(input_value, prototypes)
        m_d = nump.min(Renyi_div, axis=1)
        expand_dims = nump.expand_dims(m_d, axis=1)
        ylabel = nump.where(nump.equal(expand_dims, Renyi_div),
                            self.prototype_data_labels, nump.inf)
        ylabel = nump.min(ylabel, axis=1)
        print(ylabel)
        return ylabel

    # do_processing function
    def do_processing(self, data_in, data_labels, learning_rate, epochs):
        """
        
        Main function to train the algorithm

        :param data_in: A m x n matrix of distance.
        :param data_labels: A m-dimensional vector containing the labels for each datapoint.
        :param learning_rate: Learning rate (step size).
        :param epochs: The maximum number of optimization iterations.
        :return: A n-dimensional updated prototype vector.
        """
        xx = 0
        normalized_data = self.normalize_data(data_in)
        prototype_l, prototypes = self.create_data_prototype(normalized_data, data_labels,
                                                             self.class_prototype)
        all_errors = nump.array([])
        plt.subplots(8, 8)
        for i in range(epochs):

            Renyi_div = self.Renyi_Divergence(normalized_data, prototypes)

            d_plus, w_plus, w_plus_index = self.calculate_d_plus(data_labels,
                                                                 prototype_l,
                                                                 prototypes,
                                                                 Renyi_div)

            d_minus, w_minus, w_minus_index = self.calculate_d_minus(data_labels,
                                                                     prototype_l,
                                                                     prototypes,
                                                                     Renyi_div)

            classifier = ((d_plus - d_minus) / (d_plus + d_minus))

            sum_change_in_w_plus, unique_w_plus_index = self.change_in_w_plus(
                normalized_data, learning_rate, classifier,
                w_plus, w_plus_index, d_plus, d_minus)
            update_w_p = nump.add(nump.squeeze(
                prototypes[unique_w_plus_index]), sum_change_in_w_plus)
            nump.put_along_axis(prototypes, unique_w_plus_index,
                                update_w_p, axis=0)

            sum_in_w_m, w_minus_idx = self.change_in_w_minus(
                normalized_data, learning_rate, classifier,
                w_minus, w_minus_index, d_plus, d_minus)
            update_w_m = nump.subtract(nump.squeeze(
                prototypes[w_minus_idx]), sum_in_w_m)
            nump.put_along_axis(
                prototypes, w_minus_idx, update_w_m, axis=0)

            err = nump.sum(self.sigmoid_calc(classifier), axis=0)
            error_change = 0

            if (i == 0):
                error_change = 0

            else:
                error_change = all_errors[-1] - err

            all_errors = nump.append(all_errors, err)
            print("Epoch : {}, error : {} errors change : {}".format(
                i + 1, err, error_change))

            plt.subplot(1, 2, 1)
            self.plot(normalized_data, data_labels, prototypes, prototype_l)
            plt.subplot(1, 2, 2)
            plt.plot(nump.arange(i + 1), all_errors, marker="d")
            plt.pause(2)

        plt.show()
        #Finding the accuracy score in percentage.
        accuracy = nump.count_nonzero(d_plus < d_minus)
        acc = accuracy / len(d_plus) * 100
        print("accuracy = {}".format(acc))
        self.new_proto_types = prototypes
        return self.new_proto_types

    # plot  data
    def plot(self, data_in, data_labels, prot_types, prot_labels):
        """
        
        Visualizing the data and prototypes in a scatter plot.

        :param data_in: A n x m matrix of datapoints.
        :param data_labels: A n-dimensional vector containing the labels for each datapoint.
        :param prot_types: A n x m matrix of prototyes of each class.
        :param prot_labels: A n-dimensional vector containing the labels for each prototype.
        
        """
        plt.scatter(data_in[:, 0], data_in[:, 2], c=data_labels,
                    cmap='viridis')
        plt.scatter(prot_types[:, 0], prot_types[:, 2], c=prot_labels,
                    s=60, marker='D', edgecolor='k')

if __name__ == '__main__':
    class_no_of_prototypes = 3
    data_in = dataset().data
    data_label = dataset().target
    epochs = 3
    rate_of_learning = 0.01

    glvq_obj = GLVQ(class_no_of_prototypes)

    #Splitting the dataset into training and testing dataset

    X_train, X_test, y_train, y_test = train_test_split(data_in,
                                                        data_label,
                                                        test_size=0.3,
                                                        random_state=42)

    glvq_obj.do_processing(X_train, y_train, rate_of_learning, epochs)

    y_predict = glvq_obj.data_predictor(X_test)
