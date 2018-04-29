import videogui_start
# start GUI
# null file implies NOT to capture and save to disk to protect privacy
video_record = '/dev/null'
druid_gui = videogui_start.DruidGUI('Druid Project', video_record)
