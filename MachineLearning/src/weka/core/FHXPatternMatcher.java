/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.core;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Administrator
 */
public class FHXPatternMatcher {
    private static String m_trainFileName = null;
    private static Instances m_trainData = null;
    private static int m_classIndex;

    private static String m_selectedBufferFileName = null;

    private static Pattern p1 = Pattern.compile("EvalMethod_SearchMethod:([^\\s]*)");
    private static Pattern p2 = Pattern.compile("Selected\\sattributes:\\s([^\\s]*)\\s:");

    public static void main(String[] args) throws FileNotFoundException
    {
        String[] options = (String[])args.clone();
        try
        {
         m_trainFileName = Utils.getOption('T', options);
         File trainFile= new File(m_trainFileName);
         m_trainFileName = trainFile.getAbsolutePath();
         DataSource source = new DataSource(m_trainFileName);
         m_trainData = source.getDataSet();

         m_selectedBufferFileName = Utils.getOption('S', options);  
         File selectedBufferFile = new File(m_selectedBufferFileName);
         m_selectedBufferFileName = selectedBufferFile.getAbsolutePath();
         m_classIndex = (m_trainData.classIndex() == -1 ? (m_trainData.numAttributes() -1) : m_trainData.classIndex());
         AppendwordTotxt();
        }
        catch(Exception e) { e.printStackTrace();}
    }
    private static void AppendwordTotxt()
    {
        StringBuffer sb = new StringBuffer();
        String line =null;
        Matcher m1 =null, m2=null;
        boolean myflag = false;
        String m_NewFilename=null,m_FileData=null;

        try {
            FileReader fr = new FileReader(m_selectedBufferFileName);
            BufferedReader br = new BufferedReader(fr);
            while((line = br.readLine()) != null)
            {
               if(myflag == false ){
                   m1= p1.matcher(line);
                   if(m1.find()){
                       myflag = true;sb.append(line+"\n");
                       m_NewFilename = m1.group(1).concat(".arff");
                   }
               }
               else{
                   m2 = p2.matcher(line);
                   if(m2.find()){
                       myflag = false;sb.append(line+"\n\n");
                       m_FileData = m2.group(1);
                       CreateReducedData(m_NewFilename, m_FileData);
                   }
               }
            }
            FileWriter fw = new FileWriter(m_selectedBufferFileName, true);//appending
            fw.write("\n\n");
            fw.write(sb.toString());
            fw.close();
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }
    private static void CreateReducedData(String NewFileName,String selected)
    {
        String[] selectedIndexsStr;
        int[] selectedIndexs;
        selectedIndexsStr = selected.split(",");
        selectedIndexs = new int[selectedIndexsStr.length + 1];
        for(int i = 0; i < selectedIndexsStr.length; i++)
        {    selectedIndexs[i] = Integer.valueOf(selectedIndexsStr[i]) -1 ;}
        selectedIndexs[selectedIndexsStr.length] = m_classIndex;
        try{
         Instances reduceData;
         Remove attributeFilter = new Remove();
         attributeFilter.setAttributeIndicesArray(selectedIndexs);
         attributeFilter.setInvertSelection(true);
         attributeFilter.setInputFormat(m_trainData);
         reduceData = Filter.useFilter(m_trainData, attributeFilter);
         reduceData.setRelationName(NewFileName);
         MySave.SaveInstances(reduceData, NewFileName);
        }
        catch(Exception e){e.printStackTrace();}
    }
}
