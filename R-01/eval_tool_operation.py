import os 
import subprocess

class Eval:
    def __init__(self, eval_tool):
        self.TOOL_PATH = './eval'
        self.data_dir = '.output' # this floder have res ref
        self.run_cmd = f'{self.TOOL_PATH}/OpenDeid {self.data_dir} {self.data_dir}'
        
        self.input_res_path = None
        self.input_ref_path = None
    
            
    def set_res_path(self, res_path):
        self.input_res_path = res_path
        if not os.path.exists(os.path.join(self.data_dir, 'res')):
            os.makedirs(os.path.join(self.data_dir, 'res'))
        with open(res_path, 'r', encoding='utf-8') as f1:
            with open(os.path.join(self.data_dir, 'res', 'answer.txt'), 'w', encoding='utf-8') as f2:
                f2.write(f1.read())
    
    def set_ref_path(self, ref_path):
        self.input_ref_path = ref_path
        if not os.path.exists(os.path.join(self.data_dir, 'ref')):
            os.makedirs(os.path.join(self.data_dir, 'ref'))
        with open(ref_path, 'r', encoding='utf-8') as f1:
            with open(os.path.join(self.data_dir, 'ref', 'answer.txt'), 'w', encoding='utf-8') as f2:
                f2.write(f1.read())
    
    def run(self, detail=False):
        if self.input_res_path is None or self.input_ref_path is None:
            return 'Please set res and ref path'

        cmd = ''
        if detail:
            cmd = rf'{self.run_cmd} --detail'
        else:
            cmd = rf'{self.run_cmd}'
        
        cmd = cmd.replace(r'/' , r'\\')
        resault = subprocess.run(rf'{cmd}', shell=0) 
        
        # cvt scores.html to markdown
        if os.path.exists(os.path.join(self.data_dir, 'scores.html')):
            with open(os.path.join(self.data_dir, 'scores.html'), 'r', encoding='utf-8') as f1:
                with open(os.path.join(self.data_dir, 'SCORE_READ_THIS________.md'), 'w', encoding='utf-8') as f2:
                    file = f1.read().replace(r'<pre>' , r'').replace(r'</pre>' , r'')
                    f2.write(file)
        
        # return resault
    
            
if __name__ == '__main__':
    eval_tool = Eval('./eval')
    res_path = 'output/answer.txt'
    ref_path = 'data/First_Phase_Validation/answer.txt'
    
    
    
    eval_tool.set_res_path(res_path)
    eval_tool.set_ref_path(ref_path)
    sc = eval_tool.run(detail=1)
    print(sc)