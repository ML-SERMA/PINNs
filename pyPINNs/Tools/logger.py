import os


class Logger:
    def __init__(self, filepath, mode='w+', save_dir='', 
                header_main=None, header_extra=None, verbose=0):
        """
        filepath: filename (string)
        mode: file open mode ('w+', 'a', etc)
        save_dir: directory path
        header_main: list of strings for main loss columns, e.g. ['Iter','Loss', 'Loss_BC', 'Loss_PDE']
        header_extra: list of strings for error columns, e.g. ['Error_phi', 'Error_p']
        verbose: 0 for no print, 1 for printing logs to console
        """
        self.filepath = os.path.join(save_dir, filepath)
        self.mode = mode
        self.verbose = verbose

        self.header_main = header_main or ['Iter','Loss', 'Loss_BC', 'Loss_PDE']
        self.header_extra = header_extra or []
        self._init_file()

    def _init_file(self):
        file_exists = os.path.isfile(self.filepath)
        if self.mode == 'a' and file_exists:
            if self.verbose:
                print(f"{self.filepath} exists, continuing to append logs.")
        else:
            with open(self.filepath, 'w') as f:
                self._write_header(f)

    def _write_header(self, file):
        base_cols = []
        all_cols = base_cols + self.header_main + self.header_extra
        file.write(" ".join(all_cols) + "\n")

    def log(self, infos_main, infos_extra=None):
        """
        infos_main: list or tuple of loss values corresponding to header_loss
        infos_extra: list or tuple of error values corresponding to header_errors
        """
        with open(self.filepath, 'a') as f:
            line = ""
            for loss_val in infos_main:
                line += f" {loss_val:12g}" # 12.g
            if infos_extra:
                for err in infos_extra:
                    line += f" {err:<12.5e}"
            line += "\n"
            f.write(line)

        if self.verbose:
            log_str = f"==>:"
            log_str += ", ".join(f"{name} = {val:12g}" for name, val in zip(self.header_main, infos_main))
            if infos_extra:
                log_str += "; " + ", ".join(f"{name} = {val:<12.5e}" for name, val in zip(self.header_extra, infos_extra))
            print(log_str)








class LoggerOneGroup:
    def __init__(self, filepath, mode='w+', save_dir='', has_phi=True, has_p=False):
        self.filepath = os.path.join(save_dir, filepath)
        self.mode = mode
        self.has_phi = has_phi
        self.has_p = has_p
        self._init_file()

    def _init_file(self):
        flag = not os.path.isfile(self.filepath) # file does not exist 
        with open(self.filepath, self.mode) as f:
            if self.mode == 'a':
                if flag:
                    self._write_header(f)
                else:
                    print(f"{self.filepath} exists, continuing to append logs.")
            else:
                self._write_header(f)

    def _write_header(self, file):
        if self.has_phi and self.has_p:
            file.write(f"Iter Loss Loss_BC Loss_PDE Error_phi Error_p\n")
        elif self.has_phi:
            file.write(f"Iter Loss Loss_BC Loss_PDE Error_phi\n")
        else:
            file.write(f"Iter Loss Loss_BC Loss_PDE\n")

    def log(self, iter_num, loss, loss_bc, loss_f, error_phi=None, error_p=None, verbose=0):
        with open(self.filepath, 'a') as f:
            if error_phi is not None and error_p is not None:
                f.write(f"{iter_num:6d} {loss:12f} {loss_bc:12f} {loss_f:12f} {error_phi:<12.5f} {error_p:<12.5f}\n")
            elif error_phi is not None:
                f.write(f"{iter_num:6d} {loss:12f} {loss_bc:12f} {loss_f:12f} {error_phi:<12.5f}\n")
            else:
                f.write(f"{iter_num:6d} {loss:12f} {loss_bc:12f} {loss_f:12f}\n")

        if verbose == 1:
            log_str = f"{iter_num:6d}: Loss = {loss:12.5f}"
            if error_phi is not None:
                log_str += f"; Error_phi = {error_phi:<12.5f}"
            print(log_str)