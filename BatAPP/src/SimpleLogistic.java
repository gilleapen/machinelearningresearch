import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

public class SimpleLogistic extends JFrame implements ActionListener
{
    protected JButton m_OpenFileBut;
    protected JFileChooser m_FileChooser;
    protected JTextField m_FileDirectory;
    protected String m_ResultFileName = null;

    protected JLabel[] m_Labels_A = new JLabel[5];
    protected JLabel[] m_Labels_B = new JLabel[5];
    protected JLabel[] m_Labels_C = new JLabel[5];
    protected JTextField[] m_TextFields_A = new JTextField[5];
    protected JTextField[] m_TextFields_B = new JTextField[5];
    protected JTextField[] m_TextFields_C = new JTextField[5];

    protected JButton m_OKBut;
    protected JButton m_LookBut;

    protected StringBuffer m_sb;

    public SimpleLogistic()
    {
        m_OpenFileBut = new JButton("Open file...");
        m_OpenFileBut.setToolTipText("Open a set of instances from a file");
        m_OpenFileBut.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent e) { setFile();}});

        m_FileDirectory = new JTextField("C:\\train\\heart-statlog.arff",28);

        JPanel p1 = new JPanel();
        p1.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
        p1.setLayout(new FlowLayout(FlowLayout.LEFT,5,5));
        p1.add(m_OpenFileBut);
        p1.add(m_FileDirectory);
        add(p1,BorderLayout.NORTH);

        JPanel p2 = new JPanel();
        p2.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
        p2.setLayout(new GridLayout(4, 6, 5, 5));

        m_Labels_A[0] = new JLabel("NumofIter:");
        m_Labels_B[0] = new JLabel("          Offset:");
        m_Labels_C[0] = new JLabel("          Times:");
        m_TextFields_A[0] = new JTextField("0",5);
        m_TextFields_B[0] = new JTextField("0",5);
        m_TextFields_C[0] = new JTextField("1",5);

        p2.add(m_Labels_A[0]);
        p2.add(m_TextFields_A[0]);
        p2.add(m_Labels_B[0]);
        p2.add(m_TextFields_B[0]);
        p2.add(m_Labels_C[0]);
        p2.add(m_TextFields_C[0]);

        m_Labels_A[1] = new JLabel("MaxofIter:");
        m_Labels_B[1] = new JLabel("          Offset:");
        m_Labels_C[1] = new JLabel("          Times:");
        m_TextFields_A[1] = new JTextField("500",5);
        m_TextFields_B[1] = new JTextField("200",5);
        m_TextFields_C[1] = new JTextField("10",5);

        p2.add(m_Labels_A[1]);
        p2.add(m_TextFields_A[1]);
        p2.add(m_Labels_B[1]);
        p2.add(m_TextFields_B[1]);
        p2.add(m_Labels_C[1]);
        p2.add(m_TextFields_C[1]);

        m_Labels_A[2] = new JLabel("HeuristicStop:");
        m_Labels_B[2] = new JLabel("          Offset:");
        m_Labels_C[2] = new JLabel("          Times:");
        m_TextFields_A[2] = new JTextField("50",5);
        m_TextFields_B[2] = new JTextField("10",5);
        m_TextFields_C[2] = new JTextField("5",5);

        p2.add(m_Labels_A[2]);
        p2.add(m_TextFields_A[2]);
        p2.add(m_Labels_B[2]);
        p2.add(m_TextFields_B[2]);
        p2.add(m_Labels_C[2]);
        p2.add(m_TextFields_C[2]);

        m_Labels_A[3] = new JLabel("WeightTrim:");
        m_Labels_B[3] = new JLabel("          Offset:");
        m_Labels_C[3] = new JLabel("          Times:");
        m_TextFields_A[3] = new JTextField("0.0",5);
        m_TextFields_B[3] = new JTextField("0.0",5);
        m_TextFields_C[3] = new JTextField("1",5);

        p2.add(m_Labels_A[3]);
        p2.add(m_TextFields_A[3]);
        p2.add(m_Labels_B[3]);
        p2.add(m_TextFields_B[3]);
        p2.add(m_Labels_C[3]);
        p2.add(m_TextFields_C[3]);

        this.setDefaultCloseOperation(EXIT_ON_CLOSE);

        add(p2,BorderLayout.CENTER);

        JPanel p3 = new JPanel();
        p3.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
        p3.setLayout(new FlowLayout(FlowLayout.CENTER,5,5));

        m_OKBut = new JButton("生成批处理文件并执行文件");
        m_OKBut.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent e) { setBatAndRun();}});
        p3.add(m_OKBut);

        m_LookBut = new JButton("查看运行结果");
        m_LookBut.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent e) {try {
                    OpenBufferFile();
                } catch (Exception ex) {
                    Logger.getLogger(SimpleLogistic.class.getName()).log(Level.SEVERE, null, ex);
                }
}});
        m_LookBut.setEnabled(false);
        p3.add(m_LookBut);

        add(p3,BorderLayout.SOUTH);
    }


    public void setFile() {

        m_FileChooser = new JFileChooser(new File(System.getProperty("user.dir")));
        m_FileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        m_FileChooser.setMultiSelectionEnabled(false);
        int returnVal = m_FileChooser.showOpenDialog(this);
        if( returnVal == JFileChooser.APPROVE_OPTION)
        {
            m_FileDirectory.setText(m_FileChooser.getCurrentDirectory().toString() +"\\"+ m_FileChooser.getSelectedFile().getName());
        }
  }

    public void setBatAndRun()
    {
       m_sb  = new StringBuffer("echo off \nset path=%path%;C:\\WINDOWS\\system32;");

       m_sb.append("\nset path=%path%;C:\\Program Files\\Java\\jdk1.6.0_14\\bin");
       m_sb.append("\nset path=%path%;C:\\Program Files\\Java\\jdk1.6.0_14\\jre\\bin");
       m_sb.append("\nset path=%path%;C:\\Program Files\\Java\\jre1.6.0_07\\bin");
       m_sb.append("\nset path=%path%;C:\\Program Files\\Java\\jre6\\bin");

       m_sb.append("\nset path=%path%;D:\\Program Files\\Java\\jdk1.6.0_14\\bin");
       m_sb.append("\nset path=%path%;D:\\Program Files\\Java\\jdk1.6.0_14\\jre\\bin");
       m_sb.append("\nset path=%path%;D:\\Program Files\\Java\\jre1.6.0_07\\bin");
       m_sb.append("\nset path=%path%;D:\\Program Files\\Java\\jre6\\bin");

       m_sb.append("\nset classpath=%classpath%;C:\\trunk\\MachineLearning\\build\\classes");
       m_sb.append("\nset classpath=%classpath%;C:\\trunk\\trunk\\MachineLearning\\build\\classes");

       m_sb.append("\nset classpath=%classpath%;D:\\trunk\\MachineLearning\\build\\classes");
       m_sb.append("\nset classpath=%classpath%;D:\\trunk\\trunk\\MachineLearning\\build\\classes \n echo on");

       String trainFileName = m_FileDirectory.getText();
       int[] NumofIter = new int[Integer.parseInt(m_TextFields_C[0].getText())];
       int[] MaxofIter = new int[Integer.parseInt(m_TextFields_C[1].getText())];
       int[] HeuristicStop = new int[Integer.parseInt(m_TextFields_C[2].getText())];
       double[] WeightTrim = new double[Integer.parseInt(m_TextFields_C[3].getText())];

       Random random = new Random();
       int min_int = Integer.parseInt(m_TextFields_A[0].getText()) - Integer.parseInt(m_TextFields_B[0].getText());
       int max_int = Integer.parseInt(m_TextFields_A[0].getText()) + Integer.parseInt(m_TextFields_B[0].getText());
       for(int n = 0; n < NumofIter.length; n++)
       {    NumofIter[n] = (int)(random.nextDouble() * (max_int - min_int)) + min_int;}

       min_int = Integer.parseInt(m_TextFields_A[1].getText()) - Integer.parseInt(m_TextFields_B[1].getText());
       max_int = Integer.parseInt(m_TextFields_A[1].getText()) + Integer.parseInt(m_TextFields_B[1].getText());
       for(int n = 0; n < MaxofIter.length; n++)
       {    MaxofIter[n] = (int)(random.nextDouble() * (max_int - min_int)) + min_int;}

       min_int = Integer.parseInt(m_TextFields_A[2].getText()) - Integer.parseInt(m_TextFields_B[2].getText());
       max_int = Integer.parseInt(m_TextFields_A[2].getText()) + Integer.parseInt(m_TextFields_B[2].getText());
       for(int n = 0; n < HeuristicStop.length; n++)
       {    HeuristicStop[n] = (int)(random.nextDouble() * (max_int - min_int)) + min_int;}

       double min_dou = Double.parseDouble(m_TextFields_A[3].getText()) - Double.parseDouble(m_TextFields_B[3].getText());
       double max_dou = Double.parseDouble(m_TextFields_A[3].getText()) + Double.parseDouble(m_TextFields_B[3].getText());
       for(int n = 0; n < WeightTrim.length; n++)
       {    WeightTrim[n] = random.nextDouble() * (max_dou - min_dou) + min_dou;}

       String CommandStart = "\njava weka.classifiers.functions.SimpleLogistic -t "+ trainFileName;
       //-t trainFileName, -I NumofIter, -M MaxofIter,-H HeuristicStop, -W WeightTrim
       String CommandLine = new String(CommandStart);
       String ResultFileName = trainFileName.substring(0,trainFileName.indexOf(".arff")) + "_Result_1.txt";
       StringBuffer sbCommands = new StringBuffer();
       m_ResultFileName = ResultFileName;
       for(int i = 0; i < NumofIter.length; i++)
           for(int j = 0; j < MaxofIter.length; j++)
               for(int k = 0; k < HeuristicStop.length; k++)
                   for(int m = 0; m < WeightTrim.length; m++)
                       {
                            CommandLine += " -I " + String.valueOf(NumofIter[i]);
                            CommandLine += " -M " + String.valueOf(MaxofIter[j]);
                            CommandLine += " -H " + String.valueOf(HeuristicStop[k]);
                            CommandLine += " -W " + String.valueOf(WeightTrim[m]);
                            CommandLine += " >> " + ResultFileName;
//                            for(int mn = 0; mn < RunCycles; mn ++)
//                            {   sbCommands.append(CommandLine);}
                            sbCommands.append(CommandLine);
                            CommandLine = new String(CommandStart);
                       }
    //perlString
    String perlString = "\nperl.exe Get_Options_Correct.pl " + ResultFileName + "|sort2.exe +8 -1 -n -r +7 -1 -n +5 -1 -n -u >> " + ResultFileName;
    sbCommands.append(perlString);
    BufferedWriter writer;
    String BatFileName = trainFileName.substring(0,trainFileName.indexOf(".arff")) + "_Result_0.bat";
    try{
    writer = new BufferedWriter(new FileWriter(BatFileName,false));
    m_sb.append(sbCommands);
    writer.write(m_sb.toString());
    writer.flush();
    writer.close();
    }
    catch(Exception e)
    { e.printStackTrace(); }
    try {
    Runtime.getRuntime().exec("cmd /c start " + BatFileName);
    // Runtime.getRuntime().exec("cmd   "+BatFileName+" 1>log1 2>log2");
    }catch(Exception e){e.printStackTrace();}

    }

    public void OpenBufferFile() throws Exception
    {
        Runtime   rt = Runtime.getRuntime();
        try{rt.exec("notepad.exe", new String[]{m_ResultFileName}); }
        catch(Exception e){ e.printStackTrace();}
    }

    public static void main(String[] args) throws Exception
    {
        SimpleLogistic example = new SimpleLogistic();
        example.setSize(600, 260);
        example.setVisible(true);

    }

    public void actionPerformed(ActionEvent e) {
        throw new UnsupportedOperationException("Not supported yet.");
    }
}

