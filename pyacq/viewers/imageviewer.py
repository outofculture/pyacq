# -*- coding: utf-8 -*-
# Copyright (c) 2016, French National Center for Scientific Research (CNRS)
# Distributed under the (new) BSD License. See LICENSE for more info.

from collections import OrderedDict

from ..core import WidgetNode, register_node_type, InputStream

from pyqtgraph.Qt import QtCore, QtGui

import numpy as np
import pyqtgraph as pg


class ImageViewer(WidgetNode):
    """
    A simple image viewer using pyqtgraph.
    """
    _input_specs = {'video': dict(streamtype='video',dtype='uint8',
                                                shape=(-1, -1, 3), compression ='',
                                                )}
    def __init__(self, **kargs):
        WidgetNode.__init__(self, **kargs)
        
        self.layout = QtGui.QHBoxLayout()
        self.setLayout(self.layout)


        self.graphiclayout = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphiclayout)
        
        #~ self.graphicsview = pg.GraphicsView()
        #~ self.layout.addWidget(self.graphicsview)
        
        #~ self.plot = pg.PlotItem()
        #~ self.graphicsview.setCentralItem(self.plot)
        #~ self.plot.getViewBox().setAspectLocked(lock=True, ratio=1)
        #~ self.plot.hideButtons()
        #~ self.plot.showAxis('left', False)
        #~ self.plot.showAxis('bottom', False)
                
        #~ self.image = pg.ImageItem()
        #~ self.plot.addItem(self.image)
    
    def _configure(self, num_video=1, nb_column=4, **kargs):
        self.num_video = num_video
        self.nb_column = nb_column
        
        if num_video == 1:
            pass
        else:
            self.inputs = OrderedDict()
            for i in range(self.num_video):
                stream_spec = dict(streamtype='video', dtype='uint8',)
                stream_spec['port'] = '*'
                stream_spec['shape'] = (-1, -1, 3)
                input = InputStream(spec=stream_spec)
                self.inputs[f'video{i}'] = input
   
    def _initialize(self):
        all_rates = [input.params['sample_rate'] for k, input in self.inputs.items()]
        max_rate = max(all_rates)
        self.timer = QtCore.QTimer(singleShot=False)
        self.timer.setInterval(int(1. / max_rate * 1000))
        self.timer.timeout.connect(self.poll_socket)
        
        self.plots = []
        self.images = []
        for i in range(self.num_video):
            plot = pg.PlotItem()
            r =  i  // self.nb_column
            c = i % self.nb_column
            self.graphiclayout.addItem(plot, r, c)
            self.plots.append(plot)
            #~ self.graphicsview.setCentralItem(self.plot)
            plot.getViewBox().setAspectLocked(lock=True, ratio=1)
            plot.hideButtons()
            plot.showAxis('left', False)
            plot.showAxis('bottom', False)

            image = pg.ImageItem()
            plot.addItem(image)
            self.images.append(image)
        

    def _start(self):
        self.timer.start()

    def _stop(self):
        self.timer.stop()
    
    def _close(self):
        pass
    
    def poll_socket(self):
        for i, input in enumerate(self.inputs.values()):
            event = input.socket.poll(0)
            if event != 0:
                while input.socket.poll(0)>0:
                    index, data = input.recv()
                data = data[::-1,:,:]
                data = data.swapaxes(0,1)
                self.images[i].setImage(data)


register_node_type(ImageViewer)

"""
import vispy
import vispy.scene


class ImageViewer(WidgetNode):
    _input_specs = {'video' : dict(streamtype = 'video',dtype = 'uint8',
                                                shape = (-1, -1, 3), compression ='',
                                                ),
                                }
    def __init__(self, **kargs):
        WidgetNode.__init__(self, **kargs)
        
        self.layout = QtGui.QHBoxLayout()
        self.setLayout(self.layout)
        
        self.canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        self.layout.addWidget(self.canvas.native)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = vispy.scene.PanZoomCamera(aspect=1)
        #~ self.view.camera = vispy.scene.MagnifyCamera(aspect=1)
        
        
    def start(self):
        self.timer.start()
        self._running = True

    def stop(self):
        self.timer.stop()
        self._running = False
    
    def close(self):
        pass
    
    def initialize(self):
        in_params = self.input.params
        img_data = np.zeros(in_params['shape']).astype(in_params['dtype'])
        self.image = vispy.scene.visuals.Image(img_data, parent=self.view.scene)
        # please luke hepl me here I do not known how to range the image in the full canvas
        self.view.camera.rect = (0,0) + tuple(in_params['shape'][:2])
        
        self.timer = QtCore.QTimer(singleShot=False)
        self.timer.setInterval(int(1./in_params['sample_rate']*1000))
        self.timer.timeout.connect(self.poll_socket)

    def configure(self, **kargs):
        pass
    
    def poll_socket(self):
        event =  self.input.socket.poll(0)
        if event != 0:
            index, data = self.input.recv()
            # this is a vertical flip
            # this should be done in GPU 
            # please help
            data = data[::-1,:,:]
            self.image.set_data(data)
            self.image.update()
"""
