'''import warnings
warnings.filterwarnings('ignore')'''
from tkinter import OptionMenu, StringVar, ttk
import tkinter as tk
from utius import *
from tkinter.constants import HORIZONTAL
from ttkthemes  import ThemedTk
from tkinter.filedialog import askdirectory
from tkinter import messagebox
from PIL import Image, ImageTk
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def open_dir():

    instText.set("loading...")
    browse_bin.destroy()

    root.directory = askdirectory() 

    pageContent=root.directory
    lApp=lacevApp()
    lApp.startLacevApp(pageContent)
    filter=Filter()
    cf = Features()

    textMsg=lApp.startDir()
    liFrame =ttk.Frame(root, width=40,height=10)
    liFrame.grid(column = 1, row = 2, padx=10,pady=10)
    scrollbar = ttk.Scrollbar(liFrame)
    text_box = tk.Text(liFrame, height=10, width=50, padx=15, pady=15)
    text_box.tag_configure("center", justify="center")
    lengthText=100/(len(textMsg)-1)
    for i in textMsg:
        text_box.insert(1.0, i)    
        progBar['value'] += lengthText
        progBar.update_idletasks()
        text_box.pack()
    scrollbar.configure(command=text_box.yview)
    scrollbar.pack(side="right", fill="y")    
    progBar['value'] = 0
    progBar.update_idletasks()
    
    filepathC=lApp.getfilepathC()
    filePathR=lApp.getfilepatR()
    classe = 0
    for pathC in filepathC:
        pathFC = filePathR+'\\'+pathC
        directory = os.listdir(pathFC)
        resu = []
        for file in directory:
            fileN, exte  = os.path.splitext(str(file))

            if(exte == '.csv' or exte == '.txt'):
                pathFileFinal = str(pathFC+'\\'+file)
                dataP = pd.read_csv(pathFileFinal, encoding = 'utf-8', 
                                        delimiter=",", 
                                        engine='c', 
                                        low_memory=False, 
                                        memory_map=True,
                                        )
                
                text_box.insert(1.0, 'Filtering '+fileN+'. Please, wait.. \n')
                text_box.pack()
                mVolt=filter.starFilter(dataP.to_numpy()[:,0].ravel(), pathC=pathC, fileN=fileN)
                
                dataP['MicroVF']=mVolt[1]
                
                #canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
                #canvas.draw()
                #canvas.get_tk_widget().grid(column=1, row=4)               
                
                
                if 'HORA' in dataP.columns:
                    Time= []
                    for i in dataP['HORA']:
                        tsplit=str(i).split(':')
                        Time.append(
                            int(str(tsplit[0]).replace(' ', '').replace('\\', '').replace('n', '').rstrip().lstrip() +str(tsplit[1]).replace('\\', '').replace('n', '').replace(' ', '').rstrip().lstrip())
                        )
                else:
                    Time= range(0, len(dataP))
                
                
                #Date= str(str(dataP['Time'][0]).split(' ')[0]).split('-')
                #Date=Date[2]+'/'+Date[1]+'/'+Date[0]
                #Start creating features
                resu = cf.startFeature(mVolt[1], 3, Time[0], classe, pathC, 58, resu, text_box, fileN, progBar)
                

                text_box.insert(1.0, 'Features calculated for '+fileN+'.\n')
                text_box.pack()
                progBar['value'] = 0
                progBar.update_idletasks()
        # when you finish reading all the files in the folder, save a csv with the class feature
        df=cf.salveDf(resu, classe)       
        if 'FileNotFoundError' in str(type(df)): print('Error saving file:'+str(df))
        else:  classe +=1 # then go back to the next folder/class
        progBar['value'] = 0
        progBar.update_idletasks()
    lApp.explore(lApp.getGfilePathFe())

def actionFunction(value):
    root.directory = askdirectory() 

    pageContent=root.directory
    lApp=lacevApp()
    msg = lApp.startLacevApp(pageContent)
    if msg['erro']:
        sgmsg = lApp.startgraph()
        if sgmsg['erro']:   
            plots=Plots()
            filepathC=lApp.getfilepathC()
            instText.set("loading...")
            browse_bin.destroy()   

            liFrame =ttk.Frame(root, width=40,height=10)
            liFrame.grid(column = 1, row = 2, padx=10,pady=10)
            scrollbar = ttk.Scrollbar(liFrame)
            text_box = tk.Text(liFrame, height=10, width=50, padx=15, pady=15)
            text_box.tag_configure("center", justify="center")
            text_box.insert(1.0, "loading...")  
            text_box.pack()
            scrollbar.configure(command=text_box.yview)
            scrollbar.pack(side="right", fill="y")  
            if value == 'Ploats':
                pass
            elif value == 'FFT-PSD':
                #Plots FFT e PSD
                fig=plots.plotFftPsd(filepathC, text_box)
                if fig:
                    messagebox.showerror('Could not save graphics')
                else:
                   lApp.explore(lApp.getfilePathG()) 

            elif value == 'TDAF':
                #Plots FFT e PSD
                fig=plots.plotTDAF(filepathC, text_box)
                if fig:
                    messagebox.showerror('Could not save graphics')
                else:
                   lApp.explore(lApp.getfilePathG()) 
            
            elif value == 'TDAFLog':
                #Plots FFT e PSD
                fig=plots.plotTDAFLog(filepathC, text_box)
                if fig:
                    messagebox.showerror('Could not save graphics')
                else:
                   lApp.explore(lApp.getfilePathG()) 
        else:
            messagebox.showerror(sgmsg['msg'])
    else:
            messagebox.showerror(msg['msg'])
    

    




#root = tk.Tk()
root =  ThemedTk(theme='darkly')
root.title('lacevApp version=alpha0.2')
root.iconbitmap('lacevapp.ico')
#root.geometry('800x600+400+200')
clicked = StringVar()
clicked.set('Ploats')
drop = OptionMenu(root, clicked, "FFT-PSD", "TDAF", "TDAFLog", command=actionFunction)
drop.config(font="Raleway", bg='green', fg='white', height=2, width=15)
drop.grid(column=0, row=0)

canvas = tk.Canvas(root, width=800, height=200).grid(columnspan=3)


logo = Image.open('logo.png')
logo = ImageTk.PhotoImage(logo)
logo_label = tk.Label(image=logo)
logo_label.image = logo
logo_label.grid(column=1, row=0)

progBar =  ttk.Progressbar(root,orient='horizontal', mode='determinate',length=500, maximum = 100)
progBar.grid(column=1, row=1)

instText = tk.StringVar()
instr = tk.Label(root, textvariable=instText, font="Raleway", bg='red', fg='white')
instText.set("lacevApp needs a lot of processing power. Therefore, it is highly recommended that you close all other programs!")
instr.grid(columnspan=2, column=0, row=2)


browse_text = tk.StringVar()
browse_bin = tk.Button(root, textvariable=browse_text, command=lambda:open_dir(), font="Raleway", bg='green', fg='white', height=2, width=15)
browse_text.set('Open Folder')
browse_bin.grid(column=2, row=2)

canvas = tk.Canvas(root, width=800, height=300).grid(columnspan=3)
root.mainloop()