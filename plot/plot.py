import matplotlib.pyplot as plt

def save_bar(names, values, output_path, x_axis, y_axis):
    '''
    Create a histogram plot
    :param names: String list of the names
    :param values: Values associated with the names
    :param output_path: Output path (string)
    :param x_axis: x-axis name
    :param y_axis: y-axis name

    '''
            
    # Set position of bar on X axis
    fig = plt.figure(figsize = (len(names)//2, 5))
 
    # creating the bar plot
    plt.bar(names, values, width = 0.4)
    
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.xticks(names)
    plt.title("Discs distribution")
    plt.savefig(output_path)