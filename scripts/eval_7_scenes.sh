#!/bin/bash
###
 # @Author: Easonyesheng preacher@sjtu.edu.cn
 # @Date: 2026-04-21 15:44:00
 # @LastEditors: Easonyesheng preacher@sjtu.edu.cn
 # @LastEditTime: 2026-05-26 15:38:30
 # @FilePath: /recon/ee_recon/third_party/MASt3R-SLAM/scripts/eval_7_scenes.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
dataset_path="/opt/data/private/datasets/NVS/7Scenes/"
datasets=(
    chess
    # fire
    # heads
    # office
    # pumpkin
    # redkitchen
    # stairs
)

no_calib=true
print_only=false
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --no-calib)
            no_calib=true
            ;;
        --print)
            print_only=true
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

if [ "$print_only" = false ]; then
    for dataset in ${datasets[@]}; do
        dataset_name="$dataset_path""$dataset"/
        if [ "$no_calib" = true ]; then
            python main.py --dataset $dataset_name --no-viz --save-as 7-scenes/no_calib/$dataset --config config/eval_no_calib.yaml
        else
            python main.py --dataset $dataset_name --no-viz --save-as 7-scenes/calib/$dataset --config config/eval_calib.yaml
        fi
    done
fi

for dataset in ${datasets[@]}; do
    dataset_name="$dataset_path""$dataset"/
    echo ${dataset_name}
    if [ "$no_calib" = true ]; then
        evo_ape tum groundtruths/7-scenes/$dataset.txt logs/7-scenes/no_calib/$dataset/$dataset.txt -as
    else
        evo_ape tum groundtruths/7-scenes/$dataset.txt logs/7-scenes/calib/$dataset/$dataset.txt -as
    fi

done
