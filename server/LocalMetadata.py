import logging
import pickle


class LocalMetadata(object):

    def __init__(self, file_name="local_metadata.pickle"):
        self.local_file_path = file_name
        self.filenames = list()

    def __enter__(self, file_name="local_metadata.pickle"):
        #self.__init__()
        self.load_local_metadata()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def get_previously_uploaded(self):
        return self.filenames

    def set_previously_uploaded(self, file):
        self.filenames.extend([file])

    def load_local_metadata(self):

        try:
            with open(self.local_file_path, "rb") as f:
                metadata = pickle.load(f)

            if "filenames" not in metadata.keys():
                self.filenames = list()
            else:
                self.filenames = metadata["filenames"]

            return True
        except FileNotFoundError as e:
            logging.log(logging.WARNING, "No metadata file found, creating one now...")
            with open(self.local_file_path, "wb") as f:
                pickle.dump({"filenames": []}, f)
        except Exception as e:
            logging.log(logging.ERROR, "WHOOPS!")

        return False

    def save_local_metadata(self):
        try:
            with open(self.local_file_path, "wb") as f:
                pickle.dump({"filenames": self.filenames}, f)
            return True
        except Exception as e:
            logging.log(logging.ERROR, e)
            return False
