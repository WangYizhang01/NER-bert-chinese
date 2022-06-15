from bert_model import *
import hydra
from hydra import utils

@hydra.main(config_path="conf", config_name='config')
def main(cfg):
    suffix = '_with_crf/' if cfg.use_crf else '/'
    model = InferNer(f'{utils.get_original_cwd()}/checkpoints' + suffix, use_crf=cfg.use_crf)
    text_list = cfg.text
    if isinstance(text_list, str):
        text_list = [text_list]
    results = model.batch_predict(text_list)
    
    print("第一个NER句子:")
    print(text_list[0])
    print('第一个NER结果:')

    for k,v in results[0].items():
        if v:
            print(v,end=': ')
            if k=='PER':
                print('Person')
            elif k=='LOC':
                print('Location')
            elif k=='ORG':
                print('Organization')
   
    
if __name__ == "__main__":
    main()
