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

public class MultilayerPerceptron extends JFrame implements ActionListener
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


    public MultilayerPerceptron()
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
        p2.setLayout(new GridLayout(5, 6, 5, 5));

        m_Labels_A[0] = new JLabel("TrainingCycles:");
        m_Labels_B[0] = new JLabel("          Offset:");
        m_Labels_C[0] = new JLabel("          Times:");
        m_TextFields_A[0] = new JTextField("500",5);
        m_TextFields_B[0] = new JTextField("200",5);
        m_TextFields_C[0] = new JTextField("5",5);

        p2.add(m_Labels_A[0]);
        p2.add(m_TextFields_A[0]);
        p2.add(m_Labels_B[0]);
        p2.add(m_TextFields_B[0]);
        p2.add(m_Labels_C[0]);
        p2.add(m_TextFields_C[0]);

        m_Labels_A[1] = new JLabel("LearningRate:");
        m_Labels_B[1] = new JLabel("          Offset:");
        m_Labels_C[1] = new JLabel("          Times:");
        m_TextFields_A[1] = new JTextField("0.3",5);
        m_TextFields_B[1] = new JTextField("0.1",5);
        m_TextFields_C[1] = new JTextField("2",5);

        p2.add(m_Labels_A[1]);
        p2.add(m_TextFields_A[1]);
        p2.add(m_Labels_B[1]);
        p2.add(m_TextFields_B[1]);
        p2.add(m_Labels_C[1]);
        p2.add(m_TextFields_C[1]);
        
        m_Labels_A[2] = new JLabel("Momentums:");
        m_Labels_B[2] = new JLabel("          Offset:");
        m_Labels_C[2] = new JLabel("          Times:");
        m_TextFields_A[2] = new JTextField("0.2",5);
        m_TextFields_B[2] = new JTextField("0",5);
        m_TextFields_C[2] = new JTextField("1",5);

        p2.add(m_Labels_A[2]);
        p2.add(m_TextFields_A[2]);
        p2.add(m_Labels_B[2]);
        p2.add(m_TextFields_B[2]);
        p2.add(m_Labels_C[2]);
        p2.add(m_TextFields_C[2]);

        m_Labels_A[3] = new JLabel("RandomSeed:");
        m_Labels_B[3] = new JLabel("          Offset:");
        m_Labels_C[3] = new JLabel("          Times:");
        m_TextFields_A[3] = new JTextField("1",5);
        m_TextFields_B[3] = new JTextField("0",5);
        m_TextFields_C[3] = new JTextField("1",5);

        p2.add(m_Labels_A[3]);
        p2.add(m_TextFields_A[3]);
        p2.add(m_Labels_B[3]);
        p2.add(m_TextFields_B[3]);
        p2.add(m_Labels_C[3]);
        p2.add(m_TextFields_C[3]);

        m_Labels_A[4] = new JLabel("HiddenNodes:");
        m_Labels_B[4] = new JLabel("          Offset:");
        m_Labels_C[4] = new JLabel("          Times:");
		m_TextFields_A[4] = new JTextField("10",5);
        m_TextFields_B[4] = new JTextField("5",5);
        m_TextFields_C[4] = new JTextField("5",5);

        p2.add(m_Labels_A[4]);
        p2.add(m_TextFields_A[4]);
        p2.add(m_Labels_B[4]);
        p2.add(m_TextFields_B[4]);
        p2.add(m_Labels_C[4]);
        p2.add(m_TextFields_C[4]);
        
        this.setDefaultCloseOperation(EXIT_ON_CLOSE);
                
//        m_Labels_A[4] = new JLabel("V_Percent(%):");
//        m_Labels_B[4] = new JLabel("          RunCycles:");
//        //m_Labels_C[4] = new JLabel("          RunResult:");
//        m_TextFields_A[4] = new JTextField("10",5);
//        m_TextFields_B[4] = new JTextField("1",5);
//        //m_TextFields_C[4] = new JTextField("unknownValue",5);
//
//        p2.add(m_Labels_A[4]);
//        p2.add(m_TextFields_A[4]);
//        p2.add(m_Labels_B[4]);
//        p2.add(m_TextFields_B[4]);
//        //p2.add(m_Labels_C[4]);
//        //p2.add(m_TextFields_C[4]);

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
                    Logger.getLogger(MultilayerPerceptron.class.getName()).log(Level.SEVERE, null, ex);
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
       int[] traincycles = new int[Integer.parseInt(m_TextFields_C[0].getText())];
       double[] learningrates = new double[Integer.parseInt(m_TextFields_C[1].getText())];
       double[] monentuns = new double[Integer.parseInt(m_TextFields_C[2].getText())];
       int[] randomseeds = new int[Integer.parseInt(m_TextFields_C[3].getText())];
       int[] HiddenNodes = new int[Integer.parseInt(m_TextFields_C[4].getText())];
       Random random = new Random();
       int min_int = Integer.parseInt(m_TextFields_A[0].getText()) - Integer.parseInt(m_TextFields_B[0].getText());
       int max_int = Integer.parseInt(m_TextFields_A[0].getText()) + Integer.parseInt(m_TextFields_B[0].getText());
       for(int n = 0; n < traincycles.length; n++)
       {    traincycles[n] = (int)(random.nextDouble() * (max_int - min_int)) + min_int;}

       double min_dou = Double.parseDouble(m_TextFields_A[1].getText()) - Double.parseDouble(m_TextFields_B[1].getText());
       double max_dou = Double.parseDouble(m_TextFields_A[1].getText()) + Double.parseDouble(m_TextFields_B[1].getText());
       for(int n = 0; n < learningrates.length; n++)
       {    learningrates[n] = random.nextDouble() * (max_dou - min_dou) + min_dou;}

       min_dou = Double.parseDouble(m_TextFields_A[2].getText()) - Double.parseDouble(m_TextFields_B[2].getText());
       max_dou = Double.parseDouble(m_TextFields_A[2].getText()) + Double.parseDouble(m_TextFields_B[2].getText());
       for(int n = 0; n < monentuns.length; n++)
       {    monentuns[n] = random.nextDouble() * (max_dou - min_dou) + min_dou;}

       min_int = Integer.parseInt(m_TextFields_A[3].getText()) - Integer.parseInt(m_TextFields_B[3].getText());
       max_int = Integer.parseInt(m_TextFields_A[3].getText()) + Integer.parseInt(m_TextFields_B[3].getText());
       for(int n = 0; n < randomseeds.length; n++)
       {    randomseeds[n] = (int)(random.nextDouble() * (max_int - min_int)) + min_int;}

       min_int = Integer.parseInt(m_TextFields_A[4].getText()) - Integer.parseInt(m_TextFields_B[4].getText());
       max_int = Integer.parseInt(m_TextFields_A[4].getText()) + Integer.parseInt(m_TextFields_B[4].getText());
       for(int n = 0; n < HiddenNodes.length; n++)
       {    HiddenNodes[n] = (int)(random.nextDouble() * (max_int - min_int)) + min_int;}

       String CommandStart = "\njava weka.classifiers.functions.NewMultilayerPerceptron -t "+ trainFileName;
       //-t trainFileName, -L learningrate, -M momentnum,-N trainingcycles, -S randomseed
       String CommandLine = new String(CommandStart);
       String ResultFileName = trainFileName.substring(0,trainFileName.indexOf(".arff")) + "_Result_1.txt";
       StringBuffer sbCommands = new StringBuffer();
       m_ResultFileName = ResultFileName;
       //int RunCycles = Integer.valueOf(m_TextFields_B[4].getText());
       for(int i = 0; i < learningrates.length; i++)
           for(int j = 0; j < monentuns.length; j++)
               for(int k = 0; k < traincycles.length; k++)
                   for(int m = 0; m < randomseeds.length; m++)
                       for(int n = 0; n < HiddenNodes.length; n++)
                       {
                            CommandLine += " -L " + String.valueOf(learningrates[i]);
                            CommandLine += " -M " + String.valueOf(monentuns[j]);
                            CommandLine += " -N " + String.valueOf(traincycles[k]);
                            CommandLine += " -S " + String.valueOf(randomseeds[m]);
                            CommandLine += " -H " + String.valueOf(HiddenNodes[n]);
                            CommandLine += " >> " + ResultFileName;
                            sbCommands.append(CommandLine);
                            CommandLine = new String(CommandStart);
                       }
    //perlString
    String perlString = "\nperl.exe Get_Options_Correct.pl " + ResultFileName + "|sort2 +10 -1 -n -r +9 -1 -n -u >> " + ResultFileName;
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
        MultilayerPerceptron example = new MultilayerPerceptron();
        example.setSize(600, 260);
        example.setVisible(true);

    }

    public void actionPerformed(ActionEvent e) {
        throw new UnsupportedOperationException("Not supported yet.");
    }
}
