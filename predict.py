from bert_model import *
import hydra
from hydra import utils


def read_data(data_path):
    with open(data_path, 'r') as f:
        text_list = json.loads(f.read().strip())
    return text_list

@hydra.main(config_path="conf", config_name='config')
def main(cfg):
    suffix = '_with_crf/' if cfg.use_crf else '/'
    model = InferNer(f'{utils.get_original_cwd()}/checkpoints' + suffix, use_crf=cfg.use_crf)
    print('model load success !')
    
    if cfg.text:
        # 若cfg.text非空，则对其进行推理，否则对cfg.infer_path进行推理
        text_list = cfg.text
    else:
        text_list = read_data(cfg.infer_path)
        
    if isinstance(text_list, str):
        text_list = [text_list]
    print('data load success. start infering...')
    results = model.batch_predict(text_list)
    
    print('infer success. saving results...')
    with open(cfg.output_file, 'w') as f1:
        for res in results:
            f1.write(json.dumps(res, ensure_ascii=False))
            f1.write('\n')
    
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
