"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2021 Jan 13
Description : Polynomial Regression
"""

# This code was adapted from course material by Jenna Wiens (UMichigan).

# python libraries
import os, sys, time

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

######################################################################
# classes
######################################################################

class Data :
    
    def __init__(self, X=None, y=None) :
        """
        Data class.
        
        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """
        
        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y
    
    def load(self, filename) :
        """
        Load csv file into X array of features and y array of labels.
        
        Parameters
        --------------------
            filename -- string, filename
        """
        
        # determine filename
        try :
            dir = os.path.dirname(__file__)
        except NameError :
            dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        f = os.path.join(dir, '..', 'data', filename)
        
        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, delimiter=",")
        
        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]
    
    def plot(self, ax=None, **kwargs) :
        """Plot data."""
        
        if 'color' not in kwargs :
            kwargs['color'] = 'b'
        
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.X, self.y, **kwargs)
        ax.set_xlabel('$x$', fontsize = 16)
        ax.set_ylabel('$y$', fontsize = 16)
        plt.show()

# wrapper functions around Data class
def load_data(filename) :
    data = Data()
    data.load(filename)
    return data

def plot_data(X, y, ax=None, **kwargs) :
    data = Data(X, y)
    data.plot(ax, **kwargs)


class PolynomialRegression() :
    
    def __init__(self, m=1, alpha=0) :
        """
        Ordinary least squares regression.
        
        Attributes
        --------------------
            coef_   -- numpy array of shape (d,)
                       estimated coefficients for the linear regression problem
            m_      -- integer
                       order for polynomial regression
            lambda_ -- float
                       regularization parameter
                       (input parameter named alpha to match sklearn)
        """
        self.coef_ = None
        self.m_ = m
        self.lambda_ = alpha
    
    
    def generate_polynomial_features(self, X) :
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,1), features
        
        Returns
        --------------------
            Phi     -- numpy array of shape (n,(m+1)), mapped features
        """
        
        n,d = X.shape
        
        ### ========== TODO : START ========== ###
        # part 2a: modify to create matrix for simple linear model
        # part 3a: modify to create matrix for polynomial model
        # professor's solution: 3 lines (yours might be longer if you do not use numpy)
        #
        # hint: use np.ones(...) and either np.append(...) or np.concatenate(...)
        #       be careful about the axis you join along (e.g. rows vs columns)

        Phi = X
        
        ### ========== TODO : END ========== ###
        
        return Phi
    
    
    def fit_SGD(self, X, y, eta=0.1,
                eps=1e-10, max_iter=int(1e6),
                verbose=False, plot=False) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares stochastic gradient descent.
        
        Parameters
        --------------------
            X        -- numpy array of shape (n,d), features
            y        -- numpy array of shape (n,), targets
            eta      -- float, step size (also known as alpha)
            eps      -- float, convergence criterion
            max_iter -- integer, maximum number of iterations
            verbose  -- boolean, for debugging purposes
            plot     -- boolean, for debugging purposes
        
        Returns
        --------------------
            self     -- an instance of self
        """
        if self.lambda_ != 0 :
            raise Exception("SGD with regularization not implemented")
        
        if plot :
            fig, (ax1, ax2) = plt.subplots(1, 2)
            plt.ion()
            plt.show()
        
        Phi = self.generate_polynomial_features(X)  # map features
        n,d = Phi.shape
        eta_input = eta
        self.coef_ = np.zeros(d)                    # coefficients
        err_list  = np.zeros((max_iter,1))          # errors per iteration
        
        # SGD loop
        for t in range(max_iter) :
            ### ========== TODO : START ========== ###
            # part 2i: update step size
            # 
            # change the default eta in the function signature to 'eta=None'
            # and update the line below to your learning rate function
            if eta_input is None :
                eta = None # change this line
            else :
                eta = eta_input
            ### ========== TODO : END ========== ###
            
            # iterate through examples
            for i in range(n) :
                ### ========== TODO : START ========== ###
                # part 2d: update theta (self.coef_) using one step of SGD
                # professor's solution: 3 lines
                #
                # hint: you can simultaneously update all theta using vector math
                #       (your implementation will be longer if you update sequentially)
                
                # track error
                # hint: you cannot use self.predict(...) to make the predictions
                #       (for your own edification, see if you can figure out why)
                y_pred = y # change this line, update all predictions (not just this example)
                err_list[t] = np.sum(np.power(y - y_pred, 2)) / float(n)
                ### ========== TODO : END ========== ###
            
            # stop?
            if t > 0 and abs(err_list[t] - err_list[t-1]) < eps :
                break
            
            # debugging
            if verbose :
                cost = self.cost(X, y)
                print(f'iter {t+1}: cost = {cost}')
            if plot :
                x = np.reshape(X[:,0], (n,1))
                ax1.cla()
                plot_data(x, y, ax1)
                self.plot_regression(ax1)
                
                cost = self.cost(X, y)
                ax2.plot([t+1], [cost], 'bo')
                ax2.set_xlabel('iteration')
                ax2.set_ylabel(r'$J(\theta)$')
                
                plt.suptitle(f'iteration: {t+1}, cost: {cost:g}')
                if t == 0 :
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # account for suptitle
                    ax2.annotate('press key to continue\npress mouse to quit',
                                 xy=(0.99,0.01), xycoords='figure fraction', ha='right')
                
                plt.draw()
                keypress = plt.waitforbuttonpress(0) # True if key, False if mouse
                if not keypress :
                    plot = False
        
        print(f'number of iterations: {t+1}')
        
        return self
    
    
    def fit(self, X, y) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using the closed form solution.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
                
        Returns
        --------------------
            self    -- an instance of self
        """
        
        Phi = self.generate_polynomial_features(X) # map features
        n,d = Phi.shape
        
        ### ========== TODO : START ========== ###
        # part 2f: update theta (self.coef_) using closed-form solution
        # part 4a: include L_2 regularization
        # professor's solution: 6 lines
        #
        # hint: use np.dot(...) and np.linalg.pinv(...)

        # to aid in grading
        # please put your code without regularization before this line
        # and your code with regularization after this line
        # block comment your "before" code if you implemented regularization
        
        ### ========== TODO : END ========== ###
    
    
    def predict(self, X) :
        """
        Predict output for X.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
        
        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None :
            raise Exception("Model not initialized. Perform a fit first.")
        
        Phi = self.generate_polynomial_features(X) # map features
        
        ### ========== TODO : START ========== ###
        # part 2b: predict y
        # professor's solution: 1 line
        
        y = None
        ### ========== TODO : END ========== ###
        
        return y
    
    
    def cost(self, X, y) :
        """
        Calculates the objective function.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            cost    -- float, objective J(theta)
        """
        ### ========== TODO : START ========== ###
        # part 2c: compute J(theta)
        # professor's solution: 2 lines
        
        cost = 0
        ### ========== TODO : END ========== ###
        return cost
    
    
    def rms_error(self, X, y) :
        """
        Calculates the root mean square error.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            error   -- float, RMSE
        """
        ### ========== TODO : START ========== ###
        # part 3b: compute RMSE
        # professor's solution: 3 lines
        
        error = 0
        ### ========== TODO : END ========== ###
        return error
    
    
    def plot_regression(self, ax=None, xmin=0, xmax=1, n=100, **kwargs) :
        """Plot regression line."""
        if 'color' not in kwargs :
            kwargs['color'] = 'r'
        if 'linestyle' not in kwargs :
            kwargs['linestyle'] = '-'
        
        if ax is None :
            xmax = n
        else :
            xmax = ax.get_xlim()[1]
        
        X = np.reshape(np.linspace(0, xmax, n), (n,1))
        y = self.predict(X)
        ax.plot(X, y, **kwargs)
        
        title = rf'{self.coef_[0]:.3f}'
        for i in range(1, self.m_+1) :
            title += rf' + {self.coef_[i]:.3f} $x^{i:d}$'
        ax.set_title(title)
        
        plt.show()


######################################################################
# main
######################################################################

def main() :
    # toy data
    X = np.array([[1],
                  [2],
                  [3]])         # shape (n,d) = (3L,1L)
    y = np.array([4,5,6])       # shape (n,) = (3L,)
    coef = np.array([1,3])      # shape (d+1,) = (2L,), 1 extra for bias
    
    # load data
    train_data = load_data('regression_train.csv')
    test_data = load_data('regression_test.csv')
    
    
    
    #========================================
    # visualizations
    print('Visualizing data...')
    
    # part 1a
    train_data.plot(color='b')
    test_data.plot(color='r')
    
    print()
    
    
    
    #========================================
    # linear regression
    print('Investigating linear regression...')
    
    # model
    model = PolynomialRegression()
    
    # test part 2a -- soln: [[1 1]
    #                        [1 2]
    #                        [1 3]]
    print(f'features:\n{model.generate_polynomial_features(X)}')
    
    # test part 2b -- soln: [4 7 10]
    model.coef_ = coef
    print(f'predictions: {model.predict(X)}')
    
    # test part 2c -- soln: 10
    print(f'cost: {model.cost(X, y)}')
    
    # test part 2d
    # for eta = 0.1, soln: theta = [ 2.38405089, -2.87906028], iterations = 152
    start = time.process_time()
    model.fit_SGD(train_data.X, train_data.y, eta=0.1)
    end = time.process_time()
    print(f'sgd theta: {model.coef_} ({end-start:g} s)')
    
    ### ========== TODO : START ========== ###
    # part 2e: non-test code
    # professor's solution: 4 lines
    
    ### ========== TODO : END ========== ###
    
    # test part 2f -- soln: theta = [ 2.44640709 -2.81635359]
    start = time.process_time()
    model.fit(train_data.X, train_data.y)
    end = time.process_time()
    print(f'closed-form theta: {model.coef_} ({end-start:g} s)')
    
    ### ========== TODO : START ========== ###
    # part 2i: non-test code
    # professor's solution: 2 lines
    
    ### ========== TODO : END ========== ###

    print()
    
    
    
    #========================================
    # polynomial regression
    print('Investigating polynomial regression...')
    
    # toy data
    m = 2                                     # polynomial degree
    coefm = np.array([1,3,5]).reshape((3,))   # shape (3L,), theta_0 + theta_1 x + theta_2 x^2
    
    # test part 3a -- soln: [[1 1 1]
    #                        [1 2 4]
    #                        [1 3 9]]
    model = PolynomialRegression(m)
    print(f'features:\n{model.generate_polynomial_features(X)}')
    
    # test part 3b -- soln: 31.144823004794873
    model.coef_ = coefm
    print(f'RMSE: {model.rms_error(X, y)}')
    
    # part 3d -- RMSE vs model complexity (polynomial degree)
    m_vec = range(11)
    RMSE_train = []
    RMSE_test = []
    
    for m in m_vec :
        # learn model
        model = PolynomialRegression(m)
        model.fit(train_data.X, train_data.y)
        
        # calculate rms error
        RMSE_train.append(model.rms_error(train_data.X, train_data.y))
        RMSE_test.append(model.rms_error(test_data.X, test_data.y))
    
    plt.figure()
    plt.plot(m_vec, RMSE_train,'-or', label='Training')
    plt.plot(m_vec, RMSE_test,'-ob', label='Test')
    plt.ylim((0,3))
    plt.xlabel('$m$', fontsize = 16)
    plt.ylabel('RMSE',fontsize=16)
    plt.title('Part B')
    plt.legend(loc=2)
    plt.savefig("../plots/degree.pdf")
    #plt.show()
    plt.clf()
    
    print()
    
    
    
    #========================================
    # regularized regression
    print('Investigating regularized regression...')
    
    # test part 4a -- soln: [3 1 0]
    # note: your solution may be slightly different
    #       due to limitations in floating point representation
    #       you should get something close
    model = PolynomialRegression(m=2, alpha=1e-5)
    model.fit(X, y)
    print(f'theta: {model.coef_}')
    
    ### ========== TODO : START ========== ###
    # parts 4b: non-test code
        
    ### ========== TODO : END ========== ###
    
    print()
    
    
    
    #========================================
    print("Done!")

if __name__ == "__main__" :
    main()