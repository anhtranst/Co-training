#!/usr/bin/env bash

export CUDA_LAUNCH_BLOCKING=1

for i in 50; do
  for j in 1 2 3; do
    echo "Running with $i label/class set $j"
    python3 humaid_bertweet_optuna.py \
      --exp_name optuna-bertweet-hurricane-maria-2017-label$i-set$j \
      --dataset humanitarian9 \
      --metric_combination cv \
      --lbcl $i \
      --set_num $j \
      --hf_model_id_short gpt4o \
      --seed 1234 \
      --plm_id bert-tweet \
      --setup_local_logging \
      --optuna_trials 10 \
      --humaid hurricane_maria_2017
  done
done

for i in 50; do
  for j in 1 2 3; do
    echo "Running with $i label/class set $j"
    python3 humaid_bertweet_optuna.py \
      --exp_name optuna-bertweet-hurricane-florence-2018-label$i-set$j \
      --dataset humanitarian9 \
      --metric_combination cv \
      --lbcl $i \
      --set_num $j \
      --hf_model_id_short gpt4o \
      --seed 1234 \
      --plm_id bert-tweet \
      --setup_local_logging \
      --optuna_trials 10 \
      --humaid hurricane_florence_2018
  done
done

# # 1 - hurricane_irma_2017
# for i in 10 25 50; do
#   for j in 1 2 3; do
#     echo "Running with $i label/class set $j"
#     python3 humaid_bertweet_optuna.py \
#       --exp_name optuna-bertweet-hurricane-irma-2017-label$i-set$j \
#       --dataset humanitarian9 \
#       --metric_combination cv \
#       --lbcl $i \
#       --set_num $j \
#       --hf_model_id_short gpt4o \
#       --seed 1234 \
#       --plm_id bert-tweet \
#       --setup_local_logging \
#       --optuna_trials 14 \
#       --humaid hurricane_irma_2017
#   done
# done

# for i in 5 10 25 50; do
#   for j in 1 2 3; do
#     echo "Running with $i label/class set $j"
#     python3 humaid_bertweet_optuna.py \
#       --exp_name optuna-bertweet-canada-wildfires-2016-label$i-set$j \
#       --dataset humanitarian8 \
#       --metric_combination cv \
#       --lbcl $i \
#       --set_num $j \
#       --hf_model_id_short gpt4o \
#       --seed 1234 \
#       --plm_id bert-tweet \
#       --setup_local_logging \
#       --optuna_trials 14 \
#       --humaid canada_wildfires_2016
#   done
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_23.py \
#     --exp_name optuna-bertweet-hurricane-irma-2017-set2 \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid hurricane_irma_2017
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_31.py \
#     --exp_name optuna-bertweet-hurricane-irma-2017-set3-optuna-bertweet \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid hurricane_irma_2017
# done

# # 2 - hurricane_harvey_2017 experiments after Hongmin's clarification - 9 classes
# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_12.py \
#     --exp_name optuna-bertweet-hurricane-harvey-2017-set1 \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid hurricane_harvey_2017
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_23.py \
#     --exp_name optuna-bertweet-hurricane-harvey-2017-set2 \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid hurricane_harvey_2017
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_31.py \
#     --exp_name optuna-bertweet-hurricane-harvey-2017-set3 \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid hurricane_harvey_2017
# done

# # 3 - kerala_floods_2018 experiments after Hongmin's clarification
# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_12.py \
#     --exp_name optuna-bertweet-kerala-floods-2018-set1 \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid kerala_floods_2018
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_23.py \
#     --exp_name optuna-bertweet-kerala-floods-2018-set2 \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid kerala_floods_2018
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_31.py \
#     --exp_name optuna-bertweet-kerala-floods-2018-set3 \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid kerala_floods_2018
# done


# # 4 - hurricane_dorian_2019 experiments after Hongmin's clarification
# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_12.py \
#     --exp_name optuna-bertweet-hurricane-dorian-2019-set1 \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid hurricane_dorian_2019
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_23.py \
#     --exp_name optuna-bertweet-hurricane-dorian-2019-set2 \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid hurricane_dorian_2019
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_31.py \
#     --exp_name optuna-bertweet-hurricane-dorian-2019-set3 \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid hurricane_dorian_2019
# done


# # 5 - california_wildfires_2018 experiments after Hongmin's clarification
# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_12.py \
#     --exp_name optuna-bertweet-california-wildfires-2018-set1 \
#     --dataset humanitarian10 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid california_wildfires_2018
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_23.py \
#     --exp_name optuna-bertweet-california-wildfires-2018-set2 \
#     --dataset humanitarian10 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid california_wildfires_2018
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_31.py \
#     --exp_name optuna-bertweet-california-wildfires-2018-set3 \
#     --dataset humanitarian10 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid california_wildfires_2018
# done


# # 6 - hurricane_maria_2017 experiments after Hongmin's clarification
# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_12.py \
#     --exp_name optuna-bertweet-hurricane_maria_2017-set1 \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid hurricane_maria_2017
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_23.py \
#     --exp_name optuna-bertweet-hurricane_maria_2017-set2 \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid hurricane_maria_2017
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_31.py \
#     --exp_name optuna-bertweet-hurricane_maria_2017-set3 \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid hurricane_maria_2017
# done

# # 7 - hurricane_florence_2018 experiments after Hongmin's clarification
# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_12.py \
#     --exp_name optuna-bertweet-hurricane_florence_2018-set1 \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid hurricane_florence_2018
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_23.py \
#     --exp_name optuna-bertweet-hurricane_florence_2018-set2 \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid hurricane_florence_2018
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_31.py \
#     --exp_name optuna-bertweet-hurricane_florence_2018-set3 \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid hurricane_florence_2018
# done

# # 8 - cyclone_idai_2019 experiments after Hongmin's clarification
# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_12.py \
#     --exp_name optuna-bertweet-cyclone-idai-2019-set1 \
#     --dataset humanitarian10 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid cyclone_idai_2019
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_23.py \
#     --exp_name optuna-bertweet-cyclone-idai-2019-set2 \
#     --dataset humanitarian10 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid cyclone_idai_2019
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_31.py \
#     --exp_name optuna-bertweet-cyclone-idai-2019-set3 \
#     --dataset humanitarian10 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid cyclone_idai_2019
# done

# # 9 - canada_wildfires_2016 experiments after Hongmin's clarification
# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_12.py \
#     --exp_name optuna-bertweet-canada-wildfires-2016-set1 \
#     --dataset humanitarian8 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid canada_wildfires_2016
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_23.py \
#     --exp_name optuna-bertweet-canada-wildfires-2016-set2 \
#     --dataset humanitarian8 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid canada_wildfires_2016
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_31.py \
#     --exp_name optuna-bertweet-canada-wildfires-2016-set3 \
#     --dataset humanitarian8 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid canada_wildfires_2016
# done

# # 10 - kaikoura_earthquake_2016 experiments after Hongmin's clarification
# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_12.py \
#     --exp_name optuna-bertweet-kaikoura-earthquake-2016-set1 \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid kaikoura_earthquake_2016
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_23.py \
#     --exp_name optuna-bertweet-kaikoura-earthquake-2016-set2 \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid kaikoura_earthquake_2016
# done

# for i in 0 1 2 3; do
#   echo "Running with labeled_sample_idx=$i"
#   python3 humaid_bertweet_31.py \
#     --exp_name kaikoura-earthquake-2016-set3-optuna-bertweet \
#     --dataset humanitarian9 \
#     --metric_combination cv \
#     --labeled_sample_idx $i \
#     --hf_model_id_short gpt4o \
#     --seed 1234 \
#     --plm_id bert-tweet \
#     --setup_local_logging \
#     --humaid kaikoura_earthquake_2016
# done