# Test
for env in 5m_vs_6m 10m_vs_11m 
do 
    for model in {0..0}
    do
        for run in 0
        do
            export ENV=$env
            export MODEL=$model
            export RUN=$run
            ./sbatch-dvracer-smac.sh
            #./sbatch-dvracer-rnn-smac.sh
        done
    done
done

# Exit
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

# Inhomogeneous
for env in 2s3z 3s5z 1c3s5z 3s5z_vs_3s6z MMM2 corridor
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
