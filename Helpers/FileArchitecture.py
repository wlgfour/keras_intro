import os


class FileArchitecture:
    """
    Handles file structure for saving and loading
    Query this class to get files to save and load to. If construct_file_tree has been called, all will be initialized
    Also contains bools of whether the model has a previous saved file and activation maps
    construct_sub_dirs()
        - must have self.model_number and self.base_dir initialized
        - called if self.model_number and self.base_dir are given in __init__ call
    construct_file_tree()
        - must have called construct_sub_dirs() in order to call.
    """
    def __init__(self, model_number: str =None, base_dir: str =None, name: str ='model'):
        self.model_number = model_number  # model model number. All logs will be stored by model number
        self.base_dir = base_dir  # base dir for all logging ex: f'./log_dir/{MODEL_NUMBER}'
        self.name = name  # not necessary. Only used in save file for model weights
        self.save_file = None  # file to save model weights in
        self.act_save = None  # directory to save activation weights in
        self.tboard_dir = None  # directory to save tensorboard log files in
        self.tboard_cur_dir = None  # subdirectory of tboard_dir to store current log file
        self.tboard_hist_dir = None  # see above. past log files. must be separated for tensorboard --logdir to work
        self.load = False  # bool of whether to load model
        self.act_load = False  # bool of whether to load activation maps
        if self.model_number is not None and self.base_dir is not None:
            self.construct_sub_dirs()

    def construct_sub_dirs(self) -> None:
        self.save_file = f'{self.base_dir}/{self.name}_{self.model_number}.h5'
        self.act_save = f'{self.base_dir}/act_maps'
        self.tboard_dir = f'{self.base_dir}/tboard_{self.model_number}'
        self.tboard_cur_dir = f'{self.tboard_dir}/current'
        self.tboard_hist_dir = f'{self.tboard_dir}/history'

    def construct_file_tree(self) -> None:
        print('-- starting file management --')
        if not os.path.isdir(self.base_dir):
            print(f'    making model {self.model_number} log directory')
            os.mkdir(self.base_dir)
        if not os.path.isdir(self.tboard_dir):
            print('    making tensorboard log directory')
            os.mkdir(self.tboard_dir)
        # move past logfiles to history directory
        if not os.path.isdir(self.tboard_hist_dir):
            print('    making tensorboard history directory')
            os.mkdir(self.tboard_hist_dir)
        if not os.path.isdir(self.tboard_cur_dir):
            print('    making tensorboard current directory')
            os.mkdir(self.tboard_cur_dir)
        print('    moving tensorboard log files from current directory to history')
        log_files = [f for f in os.listdir(self.tboard_cur_dir)
                     if os.path.isfile(f'{self.tboard_cur_dir}/{f}')]
        for file in log_files:
            os.rename(f'{self.tboard_cur_dir}/{file}', f'{self.tboard_hist_dir}/{file}')
        # check for save file
        if os.path.exists(self.save_file):
            print('    found save file. setting fileArchitecture.load to True')
            self.load = True
        if os.path.isdir(self.act_save):
            print('    found act map save file. setting ACT_LOAD to true')
            self.act_load = True
        else:
            print('    making activation save file')
            os.mkdir(self.act_save)
        print('--done with file management--')
