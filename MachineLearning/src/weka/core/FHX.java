/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.core;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
/**
 *
 * @author Administrator
 */
public class FHX {
    
    private static void CreateReducedData(String trainFileName,int[] selectedAttributeSet)
            throws Exception
    {
         DataSource source = new DataSource(trainFileName);
         Instances train = source.getDataSet();
         Instances reduceData;
         Remove attributeFilter = new Remove();
         attributeFilter.setAttributeIndicesArray(selectedAttributeSet);
         attributeFilter.setInvertSelection(true);
         attributeFilter.setInputFormat(train);
         reduceData = Filter.useFilter(train, attributeFilter);

         String reducedFileName = trainFileName.substring(0,trainFileName.indexOf(".arff"));

         reducedFileName +="-";
         reducedFileName +=String.valueOf(selectedAttributeSet.length);
         String [] options = ((OptionHandler)attributeFilter).getOptions();
         for (int i = 0; i < options.length; i++) { reducedFileName += options[i].trim();}

         reduceData.setRelationName(reducedFileName);
         //windows可以长度达258个字符的绝对路径文件名（包括后缀），
         //因为盘符是硬件相关的包括三个符号，比如“c:\”或者“D:\”,所以实质是255个长度。
         if(reducedFileName.length()+".arff".length() > 258){reducedFileName = reducedFileName.substring(0, 253);}
         reducedFileName = reducedFileName.concat(".arff");
         System.out.println(reducedFileName );
         System.out.println(reducedFileName.length() );
         
         MySave.SaveInstances(reduceData, reducedFileName);
    }
    private static void CreatReducedData(String trainFileName,String selectedIndexsFileName)
            throws Exception
    {
         DataSource source = new DataSource(trainFileName);
         Instances train = source.getDataSet();
         int classIndex = (train.classIndex() == -1 ? (train.numAttributes() -1) : train.classIndex());

         BufferedReader br = new BufferedReader(new FileReader(selectedIndexsFileName));
         String lineContent;
         String[] selectedIndexsStr;
         int[] selectedIndexs;
         while( (lineContent = br.readLine()) != null)
         {
              selectedIndexsStr = lineContent.split(",");
              selectedIndexs = new int[selectedIndexsStr.length + 1];
              for(int i = 0; i < selectedIndexsStr.length; i++)
              {
                   selectedIndexs[i] = Integer.valueOf(selectedIndexsStr[i]) -1 ;
              }
              selectedIndexs[selectedIndexsStr.length] = classIndex;
              CreateReducedData(trainFileName, selectedIndexs);
         }
    }

    public static void main(String[] args)
    {
        String[] options = (String[])args.clone();
        String trainFileName;
        String selectedIndexsFileName;
        try
        {
         trainFileName = Utils.getOption('T', options);
         selectedIndexsFileName = Utils.getOption('S', options);
         File A=new File(trainFileName);
         trainFileName = A.getAbsolutePath();
         CreatReducedData(trainFileName,selectedIndexsFileName);
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }
}
