
import os

directories = [name for name in os.listdir("/media/jordan/LACIE/GradSchool/Material_Classification/minc-model/images")]
directories = sorted(directories)
print(directories)
directories = [directory for directory in directories if not directory.startswith('._')]
dir_count = 0
for dir in directories:
    if dir.startswith('._'):
        continue
    for j in range(2500):
        with open("filelist.txt", 'a') as filelist:
            number = str(j)
            while len(number) < 6:
                number = '0' + number
            filelist.write('/media/jordan/LACIE/GradSchool/Material_Classification/minc-model/images/' + dir + '/' + dir + '_' + number + '.jpg ' + str(dir_count) + '\n')
    dir_count += 1