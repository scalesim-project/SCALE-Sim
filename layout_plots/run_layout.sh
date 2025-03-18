mkdir -p benchmark_results

python3 ./scalesim/scale.py -c configs/rebuttal_test_ws.cfg -t topologies/conv_nets/Resnet18.csv -l layouts/conv_nets/resnet18.csv -p ./ >> ./benchmark_results/log_rebuttal_resnet18_ws
python3 ./scalesim/scale.py -c configs/rebuttal_test_is.cfg -t topologies/conv_nets/Resnet18.csv -l layouts/conv_nets/resnet18.csv -p ./ >> ./benchmark_results/log_rebuttal_resnet18_is
python3 ./scalesim/scale.py -c configs/rebuttal_test_os.cfg -t topologies/conv_nets/Resnet18.csv -l layouts/conv_nets/resnet18.csv -p ./ >> ./benchmark_results/log_rebuttal_resnet18_os

python3 ./scalesim/scale.py -i gemm -c configs/rebuttal_test_ws.cfg -t topologies/GEMM_mnk/vit_s.csv -l layouts/GEMM_mnk/vit_s_MK_NK.csv -p ./ >> ./benchmark_results/log_rebuttal_vit_s_ws
python3 ./scalesim/scale.py -i gemm -c configs/rebuttal_test_is.cfg -t topologies/GEMM_mnk/vit_s.csv -l layouts/GEMM_mnk/vit_s_MK_NK.csv -p ./ >> ./benchmark_results/log_rebuttal_vit_s_is
python3 ./scalesim/scale.py -i gemm -c configs/rebuttal_test_os.cfg -t topologies/GEMM_mnk/vit_s.csv -l layouts/GEMM_mnk/vit_s_MK_NK.csv -p ./ >> ./benchmark_results/log_rebuttal_vit_s_os

python3 layout_plots/layout_plot_vit.py
python3 layout_plots/layout_plot_resnet18.py