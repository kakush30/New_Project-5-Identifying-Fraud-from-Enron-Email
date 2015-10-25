import pandas

def details(filename):
    df = pandas.read_csv(filename)
    desc = df.describe()
    desc = desc.transpose()
    return desc
    
if __name__ == '__main__':
    print details('data.csv')