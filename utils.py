









def add_timestamps(trial,rcl_remote):
    # Testing
    try:
        subprocess.run(['rclone', 'copy', rcl_remote+':/BioSci-McGrath/Apps/CichlidPiData/'+trial+'/Logfile.txt', trial])
    except:
        subprocess.run(['rclone', 'copy', rcl_remote+':/McGrath/Apps/CichlidPiData/'+trial+'/Logfile.txt', trial])
    subprocess.run(['bash','pre.sh', trial])
    print(trial)

    f = pd.read_table(trial + '/Logfile_new.txt',header=None)
    videos = [f for f in glob(trial +"/*.h5", recursive=True)]
    videos.sort()
    f[0][0]

    vid = 0
    for i in range(len(f)):
        if 'PiCameraStarted'in f[0][i]:
            strt_time = f[0][i].split(",")[4][6:]
            fps = int(f[0][i].split(",")[0][28:])
            file = pd.read_hdf(videos[vid],key='p')
            file.frame /= fps
            file['time'] = pd.to_datetime(strt_time) + pd.to_timedelta(file['frame'], unit='s')
            file.to_hdf(videos[vid].split(".")[0] + "_new.h5",key='p')
            vid +=1
    print(trial)


def beh_count(trial,rcl_remote):
    # Testing
    try:
        subprocess.run(['rclone', 'copy', rcl_remote+':/BioSci-McGrath/Apps/CichlidPiData/'+trial+'/Logfile.txt', trial])
    except:
        subprocess.run(['rclone', 'copy', rcl_remote+':/McGrath/Apps/CichlidPiData/'+trial+'/Logfile.txt', trial])    print(trial)

    f = pd.read_csv(trial + '/AllClusterData.csv',index_col='Unnamed: 0')
    # videos = [f for f in glob(trial +"/*.csv", recursive=True)]
    f.drop(f.columns[:8], axis=1,inplace=True)


    f = f[f.modelAll_18_conf >= 0.9]
    f['f_count'] = 0.0

    for vid in f.videoID.unique():
        print("Processing video-" + vid)
        temp = f[f.videoID == vid]
        temp.TimeStamp = pd.to_datetime(temp.TimeStamp)
        counts = pd.read_csv(trial +"/"+vid+"_new.csv",index_col="Unnamed: 0")
        counts.time = pd.to_datetime(counts.time)

        for i in tqdm.tqdm(temp.index):
            temp.f_count[i] = counts[(counts.time >= temp.TimeStamp[i])&(counts.time <= temp.TimeStamp[i] + timedelta(seconds=int(temp.t_span[i])))]['count'].mean()

        f.loc[temp.index,'f_count'] = temp.f_count

    f.to_csv(trial + '/cluster_with_f_count.csv')

    sns.boxplot(x= 'modelAll_18_pred', y= 'f_count', data = f,showfliers=False)
    sns.stripplot(x="modelAll_18_pred", y="f_count", data=f,
              size=2, jitter=True, edgecolor="black")
    plt.show()

    print("========================================================================")
