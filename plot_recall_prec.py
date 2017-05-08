import numpy as np
import matplotlib.pyplot as plt

output_dir = "output/shell output/perim_area/ds02"

min_size = 5
recalls_5 = [float(line.rstrip('\n')) for line in open("%s/min_size_5/recall_%s.txt" % (output_dir, min_size),'r')]
precisions_5 = [float(line.rstrip('\n')) for line in open("%s/min_size_5/prec_%s.txt" % (output_dir, min_size),'r')]

min_size = 10
recalls_10 = [float(line.rstrip('\n')) for line in open("%s/min_size_10/recall_%s.txt" % (output_dir, min_size),'r')]
precisions_10 = [float(line.rstrip('\n')) for line in open("%s/min_size_10/prec_%s.txt" % (output_dir, min_size),'r')]

min_size = 15
recalls_15 = [float(line.rstrip('\n')) for line in open("%s/min_size_15/recall_%s.txt" % (output_dir, min_size),'r')]
precisions_15 = [float(line.rstrip('\n')) for line in open("%s/min_size_15/prec_%s.txt" % (output_dir, min_size),'r')]

min_size = 20
recalls_20 = [float(line.rstrip('\n')) for line in open("%s/min_size_20/recall_%s.txt" % (output_dir, min_size),'r')]
precisions_20 = [float(line.rstrip('\n')) for line in open("%s/min_size_20/prec_%s.txt" % (output_dir, min_size),'r')]

min_size = 10
eu_recalls_10 = [float(line.rstrip('\n')) for line in open("%s/EU_m_10/recall_%s.txt" % (output_dir, min_size),'r')]
eu_precisions_10 = [float(line.rstrip('\n')) for line in open("%s/EU_m_10/prec_%s.txt" % (output_dir, min_size),'r')]

min_size = 20
eu_recalls_20 = [float(line.rstrip('\n')) for line in open("%s/EU_m_20/recall_%s.txt" % (output_dir, min_size),'r')]
eu_precisions_20 = [float(line.rstrip('\n')) for line in open("%s/EU_m_20/prec_%s.txt" % (output_dir, min_size),'r')]


fig = plt.figure()
plt.plot(recalls_5,precisions_5, marker='o',linewidth=3.0, linestyle='--', color='r', label='SAD, min_size=5')
plt.plot(recalls_10,precisions_10, marker='o',linewidth=3.0, linestyle='--', color='b', label='SAD, min_size=10')
plt.plot(recalls_15,precisions_15, marker='o',linewidth=3.0, linestyle='--', color='g', label='SAD, min_size=15')
plt.plot(recalls_20,precisions_20, marker='o',linewidth=3.0, linestyle='--', color='m', label='SAD, min_size=20')
plt.plot(eu_recalls_10,eu_precisions_10, marker='s',linewidth=3.0, linestyle='--', color='c', label='EU, min_size=10')
plt.plot(eu_recalls_20,eu_precisions_20, marker='s',linewidth=3.0, linestyle='--', color='k', label='EU, min_size=20')

plt.xlabel('Recall',fontsize=18)
plt.ylabel('Precision',fontsize=18)
plt.legend()
leg = plt.gca().get_legend()
ltext  = leg.get_texts()  
llines = leg.get_lines() 
plt.setp(ltext, fontsize=18)    # the legend text fontsize
plt.setp(llines, linewidth=2.5)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

