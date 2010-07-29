/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.core;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 *
 * @author Administrator
 */
public class FHXPatternMatcherForClassify {

    private static String m_FileName;
    private static String m_Tag;
    private static Pattern p1 = Pattern.compile("OPC_");
    private static Pattern p2 = Pattern.compile("Correctly\\sClassified\\sInstances[^0-9]*[0-9]*[^0-9]*([^\\s]*)");
    private static StringBuffer sb = null;
    public static void main(String[] args) throws Exception
    {
        String[] options = (String[])args.clone();
        try
        {
            m_FileName = Utils.getOption('T', options);
            m_Tag = Utils.getOption('S', options);
            Sort();
            FileWriter fw = new FileWriter(m_FileName, true);//appending
            fw.write("\n\nClassification_Summary:\n");
            fw.write(sb.toString());
            fw.close();
        }
        catch(Exception e){e.printStackTrace();}
    }
    private static void Sort()throws FileNotFoundException
    {
        sb = new StringBuffer();
        boolean flag1 = false;
        boolean flag2 = false;
        Matcher m = null;
        p1=Pattern.compile(m_Tag);
        int myIndex = 0;
        ArrayList arraylist = new ArrayList();
        ClsPercent clsper = null;
        try
        {
            FileReader fr = new FileReader(m_FileName);
            BufferedReader br = new BufferedReader(fr);
            String line;
            while((line = br.readLine()) != null)
            {
                if(flag1 == false)
                {
                    m = p1.matcher(line);
                    if(m.find()){flag1=true;sb.append(line);sb.append("\n");}
                }
                else
                {
                    m=p2.matcher(line);
                    if(m.find())
                    {
                        if(flag2 == false){flag2 = true;}
                        else
                        {
                            flag1 = false;flag2 = false;
                            clsper = new ClsPercent();
                            clsper.setIndex(myIndex++);
                            clsper.setPercent(Double.parseDouble(m.group(1)));
                            arraylist.add(clsper);
                            sb.append(line);sb.append("\n\n");
                        }
                    }
                }
            }
            Collections.sort(arraylist, new SortByPercent());
            CreateSortedBuffer(arraylist);
       }
       catch(Exception e){e.printStackTrace();}
    }
    private static void CreateSortedBuffer(ArrayList a)
    {
        int[] indexs = new int[a.size()];
        for(int i =0;i<a.size();i++)
        { indexs[i] =((ClsPercent)a.get(i)).getIndex();}
        
    }

}
//
class ClsPercent
{
    private double m_percent;
    private int m_index;
    public double getPercent(){return m_percent;}
    public void setPercent(double per){m_percent=per;}
    public int getIndex(){return m_index;}
    public void setIndex(int index){m_index = index;}

}
class SortByPercent implements Comparator
{
        public int compare(Object o1, Object o2)
        {
            ClsPercent a = (ClsPercent)o1;
            ClsPercent b = (ClsPercent)o2;
            if(a.getPercent() > b.getPercent()){    return 0;}
            else{ return 1;}
        }
}
