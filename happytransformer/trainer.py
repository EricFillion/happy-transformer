import time
import datetime
import math
from csv import DictWriter


class Trainer:
    def __init__(self, model, model_type, tokenizer, device, runner, logger):
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.device = device
        self.runner = runner
        self.logger = logger


    def train(self, filepath, args):
        raise NotImplementedError()
        pass

    def test(self, filepath, args, output_filepath):
        raise NotImplementedError()
        pass

    def eval(self, filepath, args, output_filepath):
        raise NotImplementedError()
        pass

    def _get_train_eval_data(self, filepath):
        """
        Used for parsing data for training and evaluating (both contain labels)
        :param filepath: a string that contains the location of the data
        :return:
        """
        pass

    def _get_test_data(self, filepath):
        pass

    def _format_time(self, time):
        """
        elapsed: time in seconds
        return: time outputted in hh:mm:ss format
        """
        time_rounded = int(round((time)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=time_rounded))


    def _get_update_interval(self, count):
        """
        Determines how often to print status, given the number of cases.

        First determines how often to update for exactly 50 updates.
        Then, rounds to the nearest power of ten (10, 100, 1000 etc)


        :param count:
        :return:
        """

        x = count / 50
        order = math.floor(math.log(x, 10))

        update_interval = 10 ** order
        if update_interval == 0:
            return 1
        return update_interval

    def _print_status(self, init_time, count, total, update_interval, percentage = None):
        if count % update_interval and not count == 0:
            current_time = time.time()
            elapsed_time_string = self._format_time(current_time - init_time)

            avg_ex = (current_time - init_time) / count
            rem_time_int = avg_ex * (total - count)
            rem_time_string = self._format_time(rem_time_int)
            ending = ""
            if percentage != None:
                ending = "Correct: " + str(round(percentage, 2)*100) + "%"
            status_output = "Done: ", str(count) + "/" + str(
                total) + "  ----  Elapsed: " + elapsed_time_string + "   Estimated Remaining: " + rem_time_string +"  " + ending


            self.logger.info(status_output)

    def _output_result_to_csv(self, output_filepath, fieldnames, results):
        with open(output_filepath, 'w') as csv_file:
            csv_writer = DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            for result in results:
                csv_writer.writerow(
                    result
                )
