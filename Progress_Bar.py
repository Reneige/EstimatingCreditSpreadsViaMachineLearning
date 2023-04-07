# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 13:18:45 2023

@author: Renea
"""
from tkinter import Toplevel, ttk, messagebox


class Progress_Bar:
    ''' Creates a tKinter progress bar. Must instantiates by setting the number of steps in the bar
        you can optionally add custom final messages or progress messages. You can also change the progress 
        message with the self.change_message method
    '''

    def __init__(self, steps, final_message='Database Build Complete', message='Current Progress'):
        self.steps = steps
        self.final_message = final_message
        self.message = message

    def progress(self):
        ''' displays a progress bar as data is parsed from excel files. The two following methods are helper 
            methods for this progress bar method
        '''

        self.pb_popup = Toplevel()
        self.pb_popup.title('Progress...')
        self.pb_popup.geometry("300x100")

        self.pb = ttk.Progressbar(
            self.pb_popup,
            orient='horizontal',
            mode='determinate',
            length=300)

        self.value_label = ttk.Label(
            self.pb_popup, 
            text=self.update_progress_label()
            )
        self.value_label.grid(column=0, row=1, columnspan=2)
        self.pb.grid(column=0, row=2, columnspan=2)

    def update_progress_label(self):
        return f"{self.message}: {self.pb['value']:.1f}%"

    def progress_step(self):
        ''' triggered each time the progress bar takes a step forward '''
        self.pb['value'] += (100 /
                             self.steps)  # Note : updated after condition
        if (self.pb['value'] < 100):
            self.value_label['text'] = self.update_progress_label()
        else:
            messagebox.showinfo(message=self.final_message)
            self.pb_popup.destroy()

    def progress_jump(self):
        ''' make the progress bar leap - used after slow / long processes '''
        for _ in range(50):
            self.progress_step()

    def change_message(self, new_message):
        ''' changes the message displayed on progress bar'''
        self.message = new_message
