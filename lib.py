import configparser
import os
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from math import sqrt

def newIcon(icon):
    return QIcon(':/' + icon)

def newAction(parent, text, slot=None, shortcut=None, icon=None,
        tip=None,statustip=None, checkable=False, enabled=True):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QAction(text, parent)
    if icon is not None:
        a.setIcon(newIcon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if statustip is not None:
        a.setStatusTip(statustip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    return a
def newWidgetAction(parent,widget):
    action=QWidgetAction(parent)
    action.setDefaultWidget(widget)
    return action
def addActions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)
def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())

def getConfig(section,key):
    cf = configparser.ConfigParser()
    path = os.path.join(os.path.dirname(__file__), 'app.conf')
    cf.read(path)
    value=cf.get(section, key)
    return value


class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
