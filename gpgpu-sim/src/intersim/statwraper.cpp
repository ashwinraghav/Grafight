//a Wraper function for stats class
#include "stats.hpp"

extern "C"
void* StatCreate (char * name, double bin_size, int num_bins) {
   Stats* newstat = new Stats(NULL,name,bin_size,num_bins);
   newstat->Clear ();
   return(void *) newstat;  
}

extern "C"
void StatClear(void * st)
{
   ((Stats *)st)->Clear();
}

extern "C"
void StatAddSample (void * st, int val)
{
   ((Stats *)st)->AddSample(val);
}

extern "C"
double StatAverage(void * st) 
{
   return((Stats *)st)->Average();
}

extern "C"
double StatMax(void * st) 
{
   return((Stats *)st)->Max();
}

extern "C"
double StatMin(void * st) 
{
   return((Stats *)st)->Min();
}

extern "C"
void StatDisp (void * st)
{
   printf ("Stats for ");
   ((Stats *)st)->DisplayHierarchy();
   if (((Stats *)st)->NeverUsed()) {
      printf (" was never updated!\n");
   } else {
      printf("Min %f Max %f Average %f \n",((Stats *)st)->Min(),((Stats *)st)->Max(),StatAverage(st));
      ((Stats *)st)->Display();
   }
}

extern "C"
void StatDumptofile (void * st, FILE f)
{

}

#if 0 
int main ()
{
   void * mytest = StatCreate("Test",1,5);
   StatAddSample(mytest,4);
   StatAddSample(mytest,4);StatAddSample(mytest,4);
   StatAddSample(mytest,2);
   StatDisp(mytest);
}
#endif


