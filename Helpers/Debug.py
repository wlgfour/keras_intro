class Debug(object):
    """
    class that says whether various parts of program are debugging
    add flags to debug to toggle off
    DEBUG + 'str' means that str is off
    DEBUG + 'str' returns whether to execute block
    *args is a list of blacklisted blocks

    >>> DEBUG = Debug(True, 'train')
    >>> DEBUG + 'train'
    False
    """
    def __init__(self, debug: bool, *args: str):
        super(Debug, self).__init__()
        self._flags = list(args)
        if debug:
            self._flags.append('debug')  # if DEBUG and 'debug' means debugging
            print('----------DEBUGGING----------')

    def __add__(self, other: str):
        # returns false if debug is on and  flag in flags
        # true if debug is off or other not in flags
        debugging = 'debug' in self._flags
        flag = other in self._flags
        return not(debugging and flag)
