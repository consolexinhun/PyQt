/********************************************************************************
** Form generated from reading UI file 'main.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef MAIN_H
#define MAIN_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>
#include <appMain>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionOpen;
    QAction *actionSave;
    QAction *actionClear;
    QAction *actionQuit;
    QAction *actionRot90;
    QAction *actionThreshold;
    QAction *actionGray;
    QAction *actionCanny;
    QAction *actionPSP;
    QWidget *centralwidget;
    ImageView *sourceImg;
    QLabel *msg;
    ImageView *targetImg;
    QLabel *msg_width;
    QLabel *msg_height;
    QLabel *label_width;
    QLabel *label_height;
    QLabel *label;
    QLabel *label_2;
    QMenuBar *menubar;
    QMenu *menu;
    QMenu *menu_2;
    QMenu *menu_3;
    QStatusBar *statusbar;
    QToolBar *toolBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1104, 758);
        actionOpen = new QAction(MainWindow);
        actionOpen->setObjectName(QString::fromUtf8("actionOpen"));
        actionSave = new QAction(MainWindow);
        actionSave->setObjectName(QString::fromUtf8("actionSave"));
        actionClear = new QAction(MainWindow);
        actionClear->setObjectName(QString::fromUtf8("actionClear"));
        actionQuit = new QAction(MainWindow);
        actionQuit->setObjectName(QString::fromUtf8("actionQuit"));
        actionRot90 = new QAction(MainWindow);
        actionRot90->setObjectName(QString::fromUtf8("actionRot90"));
        actionThreshold = new QAction(MainWindow);
        actionThreshold->setObjectName(QString::fromUtf8("actionThreshold"));
        actionGray = new QAction(MainWindow);
        actionGray->setObjectName(QString::fromUtf8("actionGray"));
        actionCanny = new QAction(MainWindow);
        actionCanny->setObjectName(QString::fromUtf8("actionCanny"));
        actionPSP = new QAction(MainWindow);
        actionPSP->setObjectName(QString::fromUtf8("actionPSP"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        sourceImg = new ImageView(centralwidget);
        sourceImg->setObjectName(QString::fromUtf8("sourceImg"));
        sourceImg->setGeometry(QRect(10, 50, 531, 551));
        msg = new QLabel(centralwidget);
        msg->setObjectName(QString::fromUtf8("msg"));
        msg->setGeometry(QRect(30, 620, 741, 21));
        targetImg = new ImageView(centralwidget);
        targetImg->setObjectName(QString::fromUtf8("targetImg"));
        targetImg->setGeometry(QRect(560, 50, 531, 551));
        msg_width = new QLabel(centralwidget);
        msg_width->setObjectName(QString::fromUtf8("msg_width"));
        msg_width->setGeometry(QRect(120, 640, 121, 31));
        msg_height = new QLabel(centralwidget);
        msg_height->setObjectName(QString::fromUtf8("msg_height"));
        msg_height->setGeometry(QRect(370, 640, 121, 31));
        label_width = new QLabel(centralwidget);
        label_width->setObjectName(QString::fromUtf8("label_width"));
        label_width->setGeometry(QRect(20, 650, 72, 15));
        label_height = new QLabel(centralwidget);
        label_height->setObjectName(QString::fromUtf8("label_height"));
        label_height->setGeometry(QRect(270, 650, 72, 15));
        label = new QLabel(centralwidget);
        label->setObjectName(QString::fromUtf8("label"));
        label->setGeometry(QRect(220, 20, 72, 15));
        label_2 = new QLabel(centralwidget);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setGeometry(QRect(790, 20, 72, 15));
        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 1104, 26));
        menu = new QMenu(menubar);
        menu->setObjectName(QString::fromUtf8("menu"));
        menu_2 = new QMenu(menubar);
        menu_2->setObjectName(QString::fromUtf8("menu_2"));
        menu_3 = new QMenu(menubar);
        menu_3->setObjectName(QString::fromUtf8("menu_3"));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        MainWindow->setStatusBar(statusbar);
        toolBar = new QToolBar(MainWindow);
        toolBar->setObjectName(QString::fromUtf8("toolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, toolBar);

        menubar->addAction(menu->menuAction());
        menubar->addAction(menu_2->menuAction());
        menubar->addAction(menu_3->menuAction());
        menu->addAction(actionOpen);
        menu->addAction(actionSave);
        menu->addAction(actionClear);
        menu->addAction(actionQuit);
        menu_2->addAction(actionRot90);
        menu_2->addAction(actionThreshold);
        menu_2->addAction(actionGray);
        menu_2->addAction(actionCanny);
        menu_3->addAction(actionPSP);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "\347\273\204\347\273\207\347\227\205\347\220\206\345\233\276\345\203\217\345\210\206\345\211\262\350\275\257\344\273\266", nullptr));
        actionOpen->setText(QCoreApplication::translate("MainWindow", "\346\211\223\345\274\200", nullptr));
        actionSave->setText(QCoreApplication::translate("MainWindow", "\344\277\235\345\255\230", nullptr));
        actionClear->setText(QCoreApplication::translate("MainWindow", "\346\270\205\347\251\272", nullptr));
        actionQuit->setText(QCoreApplication::translate("MainWindow", "\351\200\200\345\207\272", nullptr));
        actionRot90->setText(QCoreApplication::translate("MainWindow", "\346\227\213\350\275\25490\302\260", nullptr));
        actionThreshold->setText(QCoreApplication::translate("MainWindow", "\344\272\214\345\200\274\345\214\226", nullptr));
        actionGray->setText(QCoreApplication::translate("MainWindow", "\347\201\260\345\272\246\345\214\226", nullptr));
        actionCanny->setText(QCoreApplication::translate("MainWindow", "\350\276\271\347\274\230\346\217\220\345\217\226", nullptr));
        actionPSP->setText(QCoreApplication::translate("MainWindow", "\346\267\261\345\272\246\347\233\221\347\235\243PSP", nullptr));
        msg->setText(QString());
        msg_width->setText(QString());
        msg_height->setText(QString());
        label_width->setText(QCoreApplication::translate("MainWindow", "\345\233\276\345\203\217\345\256\275\345\272\246\357\274\232", nullptr));
        label_height->setText(QCoreApplication::translate("MainWindow", "\345\233\276\345\203\217\351\253\230\345\272\246\357\274\232", nullptr));
        label->setText(QCoreApplication::translate("MainWindow", "\346\272\220\345\233\276\345\203\217", nullptr));
        label_2->setText(QCoreApplication::translate("MainWindow", "\347\233\256\346\240\207\345\233\276\345\203\217", nullptr));
        menu->setTitle(QCoreApplication::translate("MainWindow", "\346\226\207\344\273\266", nullptr));
        menu_2->setTitle(QCoreApplication::translate("MainWindow", "\345\233\276\345\203\217\346\223\215\344\275\234", nullptr));
        menu_3->setTitle(QCoreApplication::translate("MainWindow", "\345\233\276\345\203\217\345\210\206\345\211\262", nullptr));
        toolBar->setWindowTitle(QCoreApplication::translate("MainWindow", "toolBar", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // MAIN_H
