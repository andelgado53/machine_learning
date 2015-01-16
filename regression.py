
import numpy
import pprint
from sklearn import linear_model



variable_y  = []
variables_x = []
summary_dict = {}
        

with open('prm_hrs_oct.tsv') as f:
    variables = f.readline().strip().split('\t')
    summary_dict = { var: [] for var in variables[2:]}
    for line in f:
        line = line.strip().split('\t')
        line = [float(v) for v in line ]
        dict_line = dict(zip(variables[2:], line[2:]))
        for key in dict_line:
            summary_dict[key].append(dict_line[key])
        variables_x.append(line[2:])
        variable_y.append(float(line[1]))
        





def data_summary():

    print('variable' + '\t' + 'max value ' + '\t' + 'min value' + '\t' + 'total' + '\t' + 'AVG')
    for key in summary_dict:    
        print(key + '\t' + str(max(summary_dict[key])) + '\t' + str(min(summary_dict[key])) + '\t' + str(sum(summary_dict[key])) + '\t' + str(sum(summary_dict[key])/len(summary_dict[key])))    
    print('total customer' + '\t' + str(len(variables_x)))




def numpyfy(x, y):

    x = numpy.reshape(x, (len(x), len(x[0])))
    y = numpy.reshape(y, (len(y), 1))
    return x,y


def run_regression(x, y):

    reg = linear_model.LinearRegression()
    reg.fit(x, y)
    r_square = reg.score(x, y)
    coefs = reg.coef_

    coefs = ['{0:.10f}'.format(float(x/60.0)) for x in numpy.nditer(coefs) ]
    var = variables[2:]
    coefs = zip(var, coefs)
    print('coefficient of variables are: ')
    for var, coef in coefs:
        print(var + '\t' + coef + '\t')
    print('intercept is ' + str(reg.intercept_/60.0))
    print('R square is ' + str(r_square))
    
    



run_regression(*numpyfy(variables_x, variable_y))
data_summary()














