import os
import json
from collections import OrderedDict, Iterable

class myjson:

    def __init__(self, filename = None, parent = None, data = None, path = None):
        self.file = ''
        self.data = OrderedDict()
        self.path = ''
        
        if filename is not None: self.open(filename)
        elif parent is not None:
            self.file = parent.file
            self.data = data
            self.path = parent.path + path


    def open(self, filename):
        if not os.path.isfile(filename):
            print('Error: Cannot find file "{0}". Make sure'.format(filename))
            print('       the path is correct and the file is accessible.')
            raise IOError(filename)

        self.file = filename
        with open(self.file) as file:
            self.data = json.load(file, object_pairs_hook = OrderedDict)
            
    
    def get(self, value_path, value_type, default_value = None):
        abs_path = self.path + '.' + value_path
            
        json_data = self.data
        for path in value_path.split('.'):
            try:
                json_data = json_data[path]
            except KeyError as exc:
                if default_value is None:
                    print('Error: required JSON path not found. Operation aborted.')
                    print('       Missing path is ""'.format(abs_path))
                    raise
                else:
                    json_data = default_value
        
        if not isinstance(value_type, Iterable): value_type = [value_type]
        if type(json_data) not in value_type:
            print('Error: JSON value is of an incorrect type.')
            print('       Expected {0} but found {1}'
                .format(value_type, type(json_data)))
            print('       Invalid path is "{0}"'.format(abs_path))
            raise KeyError(value_path)

        if type(json_data) is OrderedDict:
            return myjson(parent = self, data = json_data, path = value_path)
        else:
            return json_data
            
    