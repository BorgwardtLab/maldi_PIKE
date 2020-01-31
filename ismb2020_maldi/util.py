'''
Utility functions
'''


def create_binary_label(df_resistances, antibiotic):
    '''
    Given a data frame of resistance information and the name of an
    antibiotic, creates a binary label vector. The antibiotic needs
    to be present in the data frame. Else, an error is raised.
    '''

    # TODO: check whether this conversion makes sense
    y = df_resistances[antibiotic].values
    y[y != 'R'] = 0
    y[y == 'R'] = 1

    return y.astype('int')
