# Launch continuous envs
for env in 5m_vs_6m 10m_vs_11m 27m_vs_30m 
do 
    for model in {0..1}
    do
        for run in {0..0}
        do
            export ENV=$env
            export MODEL=$model
            export RUN=$run
            ./sbatch-dvracer-smac.sh
        done
    done
done
exit


# Homogeneous
for env in 5m_vs_6m 10m_vs_11m 27m_vs_30m 2s_vs_1sc 3s_vs_5z 6h_vs_8z bane_vs_bane 2c_vs_64zg
do 
    for model in {0..1}
    do
        for run in {0..4}
        do
            export ENV=$env
            export MODEL=$model
            export RUN=$run
            ./sbatch-dvracer-smac.sh
        done
    done
done
