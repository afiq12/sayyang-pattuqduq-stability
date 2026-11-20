# Videos

The analysed videos are public YouTube videos and are **not** redistributed here.
`videos.csv` lists each video, its channel, link and the time windows used.

To re-run the video analysis, download the videos yourself (for example with `yt-dlp`) and save them in this folder as:

| File | Video |
|---|---|
| `V1_1080p.mp4` | V1, 1920x1080, 30 fps |
| `V2_720p.mp4` | V2, 1280x720, 30 fps |

Extract the V2 audio for the tempo analysis with
`ffmpeg -i V2_720p.mp4 -ac 1 -ar 22050 V2_audio.wav`.

Times in the scripts refer to the video timeline of these uploads. A re-encoded or edited copy may shift times slightly.
