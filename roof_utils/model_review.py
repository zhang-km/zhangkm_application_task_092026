"""Notebook-facing reporting for CPU model tests and the two-image sanity run."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .roof_augmentation import load_prepared


def run_model_tests(paths):
    result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],
                          cwd=paths.project_root,capture_output=True,text=True)
    print(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError('Model pipeline tests failed.')
    return True


def show_model_summary(paths):
    from .models import build_model
    model=build_model(paths,pretrained=True)
    print('Architecture: U-Net with ImageNet-pretrained ResNet18 encoder')
    print('Parameters:',f'{sum(p.numel() for p in model.parameters()):,}')
    print('Input: (B, 3, 256, 256), ImageNet-normalized RGB')
    print('Output: (B, 1, 256, 256), raw logits')
    print('Encoder feature channels: 64 -> 64 -> 128 -> 256 -> 512')
    print('Decoder channels: 128 -> 64 -> 32 -> 16 -> 16 -> 1')
    print('Encoder BatchNorm statistics fixed; decoder uses GroupNorm.')
    print('Loss: 0.5 masked BCE + 0.5 masked soft Dice loss')
    print('Primary metric: per-image roof IoU; metrics exclude invalid pixels.')
    print('This notebook explicitly uses CPU.')


def show_sanity_results(paths,run_name='sanity_two_images'):
    report=paths.reports_dir/run_name
    result=json.loads((report/'results.json').read_text())
    print('Purpose:',result['purpose'])
    print('Device:',result['device'],'| steps:',result['steps_completed'],
          '| time:',round(result['elapsed_seconds'],1),'seconds')
    print('Pass criterion: every training image reaches IoU >=',result['target_iou_each'],
          'at two consecutive evaluations.')
    print('Passed:',result['passed'],'| checkpoint reload:',result['checkpoint_reload_passed'])
    print('Untouched fold-0 validation IDs:',result['fold_validation_ids_untouched'])
    print(f"{'Image':>8} {'Initial IoU':>12} {'Final IoU':>12} {'Final Dice':>12}")
    for sid,a,b in zip(result['ids'],result['initial_metrics'],result['final_metrics']):
        print(f"{sid:>8} {a['iou']:12.4f} {b['iou']:12.4f} {b['dice']:12.4f}")
    print('\nTraining history (metrics measured on the same two training images):')
    print(f"{'Step':>6} {'Loss':>10} {'Mean IoU':>10} {'Mean Dice':>10}")
    for row in result['history']:
        print(f"{row['step']:6d} {row['loss']:10.4f} {row['mean_iou']:10.4f} {row['mean_dice']:10.4f}")
    data=np.load(report/'predictions.npz')
    for i,sid in enumerate(result['ids']):
        rgb,truth,valid=load_prepared(paths.prepared_data_dir,sid)
        logits=data['logits'][i,0]
        prediction=(logits>=0).astype(np.uint8);prediction[valid==0]=0
        overlay=rgb.copy();overlay[(prediction==1)&(valid==1)]=(0.55*rgb[(prediction==1)&(valid==1)]+0.45*np.array([255,30,30])).astype(np.uint8)
        error=np.zeros_like(rgb);error[(prediction==1)&(truth==0)&(valid==1)]=[255,60,60]
        error[(prediction==0)&(truth==1)&(valid==1)]=[40,140,255]
        canvas=Image.new('RGB',(1024,288),'white');draw=ImageDraw.Draw(canvas)
        for j,(name,array) in enumerate([('RGB',rgb),('Ground truth',truth*255),('Prediction overlay',overlay),('Errors: red FP / blue FN',error)]):
            draw.text((j*256+4,8),f'{sid} | {name}',fill='black')
            canvas.paste(Image.fromarray(array).convert('RGB'),(j*256,32))
        from IPython.display import display
        display(canvas)
    print('\nThese are training-set reconstruction scores, not generalization or test accuracy.')
    return result
