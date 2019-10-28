command="find ~/Desktop/raga-classification -name "*.mp3""
my_array=$(eval $command)

IFS=$'\n'      # Change IFS to new line
my_array=($my_array) # split to array $names


for i in "${my_array[@]}"; do echo "$i"; python3 -W ignore single_read_json.py "$i"; done
#for i in "${my_array[@]}"; do echo "$i"; done
