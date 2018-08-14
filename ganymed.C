void SetupWriter(MWriteRootFile *write, const char *name, Bool_t ismc=false) const
{
    if (!write)
        return;

    write->SetName(name);
    write->AddContainer("MHillas",        "Events");
    write->AddContainer("MHillasSrc",     "Events");
    write->AddContainer("MHillasExt",     "Events");
    write->AddContainer("MPointingPos",   "Events");
    write->AddContainer("MHillasSrcAnti", "Events", kFALSE);
    write->AddContainer("MImagePar",      "Events", kFALSE);
    write->AddContainer("MNewImagePar",   "Events", kFALSE);
    write->AddContainer("MNewImagePar2",  "Events", kFALSE);
    write->AddContainer("Hadronness",     "Events", kFALSE);
    write->AddContainer("MSrcPosCam",     "Events", kFALSE);
    write->AddContainer("MSrcPosAnti",    "Events", kFALSE);
    write->AddContainer("ThetaSquared",   "Events", kFALSE);
    write->AddContainer("OpticalAxis",    "Events", kFALSE);
    write->AddContainer("Disp",           "Events", kFALSE);
    write->AddContainer("Ghostbuster",    "Events", kFALSE);
    write->AddContainer("MEnergyEst",     "Events", kFALSE);
    write->AddContainer("MTime",          "Events", kFALSE);
    write->AddContainer("MMcEvt",         "Events", kFALSE);
    write->AddContainer("DataType",       "Events");
    write->AddContainer("FileId",         "Events");
    write->AddContainer("EvtNumber",      "Events");

    //  Added for MC adaption:

    //  write->AddContainer("Weight",         "Events");
    write->AddContainer("MMcRunHeader",         "RunHeaders", kFALSE);
    write->AddContainer("MMcCorsikaRunHeader",  "RunHeaders", kFALSE);
    write->AddContainer("MCorsikaRunHeader",    "RunHeaders", kFALSE);

	
    // ---- end add
    //    write->AddContainer("MMuonSearchPar", "Events", kFALSE);
    //    write->AddContainer("MMuonCalibPar",  "Events", kFALSE);
}

int process(MDirIter &files, const char *outfile, Float_t ra, Float_t dec, Bool_t ismc=false)
{
    /*
    MDirIter files;
    if (files.ReadList(dataset)<=0)
        return 2;
    */
    files.Sort();

    // =============================================================

    // Crab (05h34'32" 22d00'52")
    MPointingPos source("MSourcePos");
    //source.SetSkyPosition(5.5755, 22.0144); // ra[h], dec[deg]
    //source.SetSkyPosition(12.356089, 30.176897); // ra[h], dec[deg]//1218
    //source.SetSkyPosition(16.897867,39.760201);// mrk501
    //source.SetSkyPosition(11.0742, 38.2089); // ra[h], dec[deg]//mrk421

    //copy from ganymedmc:
    if (ismc)
        source.SetLocalPosition(ra, dec); // RA is ZD, DEC, is AZ!!!!
    else
        source.SetSkyPosition(ra, dec); // Copy from ganymedmc.C

    UInt_t fNumOffSourcePos = 5;

    // =============================================================

    // FIXME: If fPathIn read cuts and energy estimator from file!
    MContinue contq("MImagePar.fNumIslands>3.5"
                    "|| MNewImagePar.fNumUsedPixels<5.5",
           //         "|| MNewImagePar.fLeakage1>0.1",
                    "CutQ");

    MContinue cont0("MHillas.fSize<0"
		    "|| MHillas.GetArea > log10(MHillas.fSize)*898-1535 "
                    , "Cut0");

    // Cuts before plots
    MContinue cont2("0"
                    "|| MHillasExt.fM3Long*sign(MHillasSrc.fCosDeltaAlpha)*MGeomCam.fConvMm2Deg<-0.07"
                    "|| log10(MNewImagePar.fConc1)<-log10(MHillas.fSize)*0.57+0.13"
                    //"|| log10(MNewImagePar.fConc1)>-log10(MHillas.fSize)*0.57+0.5"
                    //"|| MHillasSrc.fDist*MGeomCam.fConvMm2Deg>log10(MHillas.fSize)*0.42+0.29"
                    "|| MHillas.GetArea*MGeomCam.fConvMm2Deg^2<0.01"
                    "|| MHillasSrc.fDist*MGeomCam.fConvMm2Deg>log10(MHillas.fSize)*0.5+0.095"
                    "|| MHillasSrc.fDist*MGeomCam.fConvMm2Deg<0.4"
                    "|| MHillasSrc.fDist*MGeomCam.fConvMm2Deg>1.2",
                    "Cut2");

    TArrayD param(11);
    // Parametrization of Disp
    // for new cleaning
    param[0]  =  1.34;  // constant
    param[8]  =  0.755; // slope
    param[9]  =  1.67972;  // leak
    param[10] =  4.86232;  // leak
    // for old cleaning
    //param[0]  =  1.34723;  // constant
    //param[8]  =  0.15214;  // slope
    //param[9]  =  0.970704; // leak
    //param[10] =  8.89826;  // leak
    // Parametrization for sign of disp (m3long, slope)
    param[5]  = -0.07;
    param[6]  =  7.2;
    param[7]  =  0.5;
    // ThetaSq-Cut
    param[1]  =  0.11;  // 0.156
    // Area-Cut
    param[2]  =  0.215468;
    param[3]  =  5.63973;
    param[4]  =  0.0836169;

    // --------------------------------------------------------------------------------

    // Setup fitter and histograms
    MAlphaFitter fit;
    fit.SetBackgroundFitMin(0.09);
    fit.SetBackgroundFitMax(0.25);
    fit.SetPolynomOrder(1);
    fit.SetSignalFunction(MAlphaFitter::kThetaSq);
    fit.SetScaleUser(1./fNumOffSourcePos); // includes fit.SetScaleMode(MAlphaFitter::kUserScale);

    // --------------------------------------------------------------------------------

    MBinning bins3(100, -0.005, 0.995, "BinningTheta",  "asin"); // MC Macro: MBinning bins3( 67, -0.005, 0.665, "BinningTheta",  "asin");
    MBinning binsD( 25,   0,      2.5, "BinningDist");
    MBinning binsC( 25,   0,      2.5, "BinningDistC");
    MBinning binsR( 60,   0,      300, "BinningRate");

    // --------------------------------------------------------------------------------

    MParList plist;
    plist.AddToList(&source);
    plist.AddToList(&fit);

    // Initialize default binnings
    plist.AddToList(&bins3);
    plist.AddToList(&binsD);
    plist.AddToList(&binsC);
    plist.AddToList(&binsR);

    MParameterI par("DataType");

   // MParameterD weight("Weight");

    plist.AddToList(&par);

   // plist.AddToList(&weight); 

    // Setup Tasklist
    MTaskList tlist;
    plist.AddToList(&tlist);

    // La Palma Magic1
    MObservatory obs;
    plist.AddToList(&obs);

    // --------------------------------------------------------------------------------

    MParameterD scale;
    scale.SetVal(1./fNumOffSourcePos);

    MHThetaSq *halphaoff = plist.FindCreateObj("MHThetaSq", "HistOff");
    halphaoff->ForceUsingSize();

    MFillH falpha(halphaoff, "", "FillHist");
  //  MFillH fenergy1("HistEOff [MHThetaSq]", "", "FillHistE"); // MC

    MFMagicCuts cuts;
    cuts.SetHadronnessCut(MFMagicCuts::kNoCut);
    cuts.SetVariables(param);

    // Cuts before thetasq
    MContinue cont1(&cuts, "Cut1");

    // Energy estimation
   // MEnergyEstimate energyest;
   //lasSrc.fDist*1.17e-2>(-6.95)) + 0.5997 *(MHillasSrc.fDist*1.17e-2 - 0.7568)^2 * (MHillasSrc.fDist*1.17e-2>0.7568) )");
 
   // MEnergyEstimate energyest;
   // energyest.SetRule("(6.99*(MHillas.fSize)^0.91 + 0.15*(MPointingPos.fZd)^2.46)");



    // Cuts before writing
    MContinue cont3("",    "Cut3");
    contq.SetAllowEmpty();
    cont0.SetAllowEmpty();
    cont1.SetAllowEmpty();
    cont2.SetAllowEmpty();
    cont3.SetAllowEmpty();

    cont1.SetInverted();

    // Filter for VsSize
    MFDataPhrase ftheta("ThetaSquared.fVal < ThetaSquaredCut.fVal", "CutT");

    // ------------- Loop Off Data --------------------
    MRead *readoffdata = NULL;
    // readoffdata.DisableAutoScheme();
    if (ismc)//added from MC
    {
        MReadMarsFile *read = new MReadMarsFile("Events");
        read->DisableAutoScheme();
        readoffdata = read;
    }
    else
    {
        MReadReports *read = new MReadReports;
        read->AddTree("Events", "MTime.", MReadReports::kMaster);
        read->AddTree("Drive",MReadReports::kRequired);
        read->AddTree("Rates",MReadReports::kRequired); // Not in original ganymed ?
        readoffdata = read;
    }

    readoffdata.AddFiles(files);

    TString fname0 = Form("%s-summary.root",  outfile);
    TString fname1 = Form("%s-analysis.root", outfile);

    MWriteRootFile dummy0(fname0.Data(), "RECREATE");
    MWriteRootFile dummy1(fname1.Data(), "RECREATE");

    MWriteRootFile *write0 = &dummy0;
    MWriteRootFile *write1 = &dummy1;
    SetupWriter(write0, "WriteAfterCut0"); //added MC, removed ismc it, as it is not used in SetupWriter
    SetupWriter(write1, "WriteAfterCut3"); //added MC,

    MWriteRootFile write2(fname1.Data(), "RECRATE");
    write2.AddContainer("MReportRates", "Rates");
    write2.AddContainer("MTimeRates",   "Rates");




    MParameterCalc setevtnum("MRawEvtHeader.fDAQEvtNumber", "SetEvtNumber");
    setevtnum.SetNameParameter("EvtNumber");

    MParameterCalc setrunnum("MRawRunHeader.GetFileID", "SetFileId");
    setrunnum.SetNameParameter("FileId");

    MFillH fill1a("MHHillasOffPre  [MHHillas]",      "MHillas",      "FillHillasPre");
    MFillH fill2a("MHHillasOffPost [MHHillas]",      "MHillas",      "FillHillasPost");
    MFillH fill3a("MHVsSizeOffPost [MHVsSize]",      "MHillasSrc",   "FillVsSizePost");
    MFillH fill3c("MHVsSizeOffTheta [MHVsSize]",     "MHillasSrc",   "FillVsSizeTheta");
    MFillH fill4a("MHHilExtOffPost [MHHillasExt]",   "MHillasSrc",   "FillHilExtPost");
    MFillH fill5a("MHHilSrcOffPost [MHHillasSrc]",   "MHillasSrc",   "FillHilSrcPost");
    MFillH fill6a("MHImgParOffPost [MHImagePar]",    "MImagePar",    "FillImgParPost");
    MFillH fill7a("MHNewParOffPost [MHNewImagePar]", "MNewImagePar", "FillNewParPost");
    //MFillH fill9a("MHEffOffTime    [MHEffectiveOnTime]", "MTime",        "FillEffOnTime");
    fill1a.SetNameTab("PreCut");
    fill2a.SetNameTab("PostCut");
    fill3a.SetNameTab("VsSize");
    fill3c.SetNameTab("CutT");
    fill4a.SetNameTab("HilExt");
    fill5a.SetNameTab("HilSrc");
    fill6a.SetNameTab("ImgPar");
    fill7a.SetNameTab("NewPar");
    //fill9a.SetNameTab("EffOffT");

    fill3c.SetFilter(&ftheta);

    //MFDataMember fbin("Bin.fVal", '>', 0);
    //fill9a.SetFilter(&fbin);

    MTaskList tlist2;
    tlist2.SetNumPasses(fNumOffSourcePos);
    fill2a.SetWeight(&scale);
    fill3a.SetWeight(&scale);
    fill3c.SetWeight(&scale);
    fill4a.SetWeight(&scale);
    fill5a.SetWeight(&scale);
    fill6a.SetWeight(&scale);
    fill7a.SetWeight(&scale);

    // How to get source position from off- and on-data?
    MSrcPosCalc scalc;
    scalc.SetMode(MSrcPosCalc::kWobble);
    scalc.SetCallback(&tlist2);

    MSrcPosCorrect scor;

    MHillasCalc hcalc;
    MHillasCalc hcalc2("MHillasCalcAnti");
    hcalc.SetAbberation(1.02);
    hcalc2.SetAbberation(1.02);
    hcalc.SetFlags(MHillasCalc::kCalcHillasSrc);
    hcalc2.SetFlags(MHillasCalc::kCalcHillasSrc);
    hcalc2.SetNameHillasSrc("MHillasSrcAnti");
    hcalc2.SetNameSrcPosCam("MSrcPosAnti");

    MSrcPosRndm srcrndm;

    tlist2.AddToList(&scalc);
    //tlist2.AddToList(&scor);
    tlist2.AddToList(&srcrndm);
    tlist2.AddToList(&hcalc);
    tlist2.AddToList(&hcalc2);
    tlist2.AddToList(&setrunnum);
    tlist2.AddToList(&setevtnum);
    tlist2.AddToList(&cuts);
    //tlist2.AddToList(&energyest); // from energy estimation
    tlist2.AddToList(write0);
    tlist2.AddToList(&cont0);  // Post/Pre shown cuts

    tlist2.AddToList(write1);

    tlist2.AddToList(&cont1);  // Calculate ThetaSq
    tlist2.AddToList(&fill2a);
    tlist2.AddToList(&ftheta); // Calculate theta Cut
    /*    if (!ismc) // added for MC
    {
        tlist2.AddToList(&falpha);
    //    tlist2.AddToList(&fenergy1);
    }*/
    std::cout << "test nach falpha"; 

    tlist2.AddToList(&fill3c); // With theta cut
    tlist2.AddToList(&cont2);  
    tlist2.AddToList(&fill3a);
    tlist2.AddToList(&fill4a);
    tlist2.AddToList(&fill5a);
    tlist2.AddToList(&fill6a);
    tlist2.AddToList(&fill7a);
    tlist2.AddToList(&cont3);




    MPointingPosCalc pcalc;
    MPointingDevCalc devcalc;

    tlist.AddToList(readoffdata);


    if (!ismc)  //added from MC ganymed
        tlist.AddToList(&pcalc,  "Drive");
    tlist.AddToList(&contq,  "Events");
    tlist.AddToList(&fill1a, "Events");
    tlist.AddToList(&tlist2, "Events");

    // by setting it here it is distributed to all consecutive tasks
    tlist.SetAccelerator(MTask::kAccDontReset|MTask::kAccDontTime);

    par.SetVal(0);
    //weight.SetVal(-1./fNumOffSourcePos); // added from MC macro


    MStatusDisplay *d = new MStatusDisplay;

    // Create and setup the eventloop
    MEvtLoop evtloop("Off");
    evtloop.SetParList(&plist);
    evtloop.SetDisplay(d);

    // Execute first analysis
    if (!evtloop.Eventloop())
        return 2;

    if (!evtloop.GetDisplay())
        return 2;


    // ------------- Loop On Data --------------------
    MRead *readondata = NULL;
    // readoffdata.DisableAutoScheme();
    if (ismc) // added from MC macro
    {
        MReadMarsFile *read = new MReadMarsFile("Events");
        read->DisableAutoScheme();
        readondata = read;
        //std::cout << "Ist aktiv";
    }
    else
    {
        MReadReports *read = new MReadReports;
        read->AddTree("Events", "MTime.", MReadReports::kMaster);
        read->AddTree("Drive",        MReadReports::kRequired);
        read->AddTree("Rates",        MReadReports::kRequired);
        read->AddTree("Humidity",     MReadReports::kRequired);
        read->AddTree("Temperatures", MReadReports::kRequired);
        readondata = read;
    }

    readondata.AddFiles(files);

    std::cout << "test nach read";

    scalc.SetMode(MSrcPosCalc::kDefault);
    scalc.SetNumRandomOffPositions(fNumOffSourcePos);

    MFillH fill1b("MHHillasOnPre  [MHHillas]",      "MHillas",      "FillHillasPre");
    MFillH fill2b("MHHillasOnPost [MHHillas]",      "MHillas",      "FillHillasPost");
    MFillH fill3b("MHVsSizeOnPost [MHVsSize]",      "MHillasSrc",   "FillVsSizePost");
    MFillH fill3d("MHVsSizeOnTheta [MHVsSize]",     "MHillasSrc",   "FillVsSizeTheta");
    MFillH fill4b("MHHilExtOnPost [MHHillasExt]",   "MHillasSrc",   "FillHilExtPost");
    MFillH fill5b("MHHilSrcOnPost [MHHillasSrc]",   "MHillasSrc",   "FillHilSrcPost");
    MFillH fill6b("MHImgParOnPost [MHImagePar]",    "MImagePar",    "FillImgParPost");
    MFillH fill7b("MHNewParOnPost [MHNewImagePar]", "MNewImagePar", "FillNewParPost");
    //MFillH fill9b("MHEffOnTime    [MHEffectiveOnTime]", "MTime",        "FillEffOnTime");
    fill1b.SetNameTab("PreCut");
    fill2b.SetNameTab("PostCut");
    fill3b.SetNameTab("VsSize");
    fill3d.SetNameTab("CutT");
    fill4b.SetNameTab("HilExt");
    fill5b.SetNameTab("HilSrc");
    fill6b.SetNameTab("ImgPar");
    fill7b.SetNameTab("NewPar");
    //fill9b.SetNameTab("EffOnT");
    fill1b.SetDrawOption("same");
    fill2b.SetDrawOption("same");
    fill3b.SetDrawOption("same");
    fill3d.SetDrawOption("same");
    fill4b.SetDrawOption("same");
    fill5b.SetDrawOption("same");
    fill6b.SetDrawOption("same");
    fill7b.SetDrawOption("same");
    //fill9b.SetFilter(&fbin);

    fill3d.SetFilter(&ftheta);

    MHThetaSq *halphaon = plist.FindCreateObj("MHThetaSq", "Hist");
    halphaon->ForceUsingSize();


    MFillH falpha2(halphaon, "", "FillHist");
    // MFillH fenergy2("HistE [MHThetaSq]", "", "FillHistE"); // from MC macro

    tlist2.SetNumPasses();

    tlist.Replace(readondata);

    MHSrcPosCam hsrcpos(kTRUE);

    MFillH fillsrc(&hsrcpos, "MSrcPosCam", "FillSrcPosCam");
    fillsrc.SetNameTab("SrcPos");

    tlist2.AddToListBefore(&fillsrc, &hcalc);

    MH3 hvs1("MPointingPos.fZd");
    MH3 hvs2("MPointingPos.fZd");
    MHVsTime hvstM("MTimeDrive.GetMjd");
    MHVsTime hvst0("MPointingPos.fZd");
    MHVsTime hvst1("MReportRates.fTriggerRate");
    MH3      hvst2("MPointingPos.fZd", "MReportRates.fTriggerRate");
    MHVsTime hvst("MReportTemperatures.GetTempMean");
    MHVsTime hvsh("MReportHumidity.GetMean");
    hvs1.SetName("ObsTime;Theta");
    hvs2.SetName("OnTime;Theta");
    hvstM.SetName("MJD");
    hvst0.SetName("ZdTime");
    hvst1.SetName("Rate");
    hvst2.SetName("RateZd;Theta;Rate");
    hvst.SetName("CamTemp");
    hvsh.SetName("CamHum");
    hvs1.SetTitle("Observation time vs. Zenith Angle;\\Theta [\\circ];T_{on} [s]");
    hvs2.SetTitle("On-Time vs. Zenith Angle;\\Theta [\\circ];T_{on} [s]");
    hvstM.SetTitle("MJD vs time;Time;MJD");
    hvst0.SetTitle("Zenith distance vs time;Time;Zd [\\circ]");
    hvst1.SetTitle("Camera trigger rate vs time;Time;Rate [Hz]");
    hvst2.SetTitle("Camera trigger rate vs Zd; Zd [\\circ];Rate [Hz]");
    hvst.SetTitle("Average sensor compartment temperature;Time;T [\\circ C]");
    hvsh.SetTitle("Average camera humidity;Time;Hum [%]");

    MFillH fillvs1(&hvs1, "", "FillObsTime");
    MFillH fillvs2(&hvs2, "", "FillOnTime");
    MFillH fillvstM(&hvstM, "MTime", "FillMjdVsTime");
    MFillH fillvst0(&hvst0, "MTimeDrive", "FillZdVsTime");
    MFillH fillvst1(&hvst1, "MTimeRates", "FillRate");
    MFillH fillvst2(&hvst2, "", "FillRateVsZd");
    MFillH fillvst(&hvst, "MTimeTemperatures", "FillCamTemp");
    MFillH fillvsh(&hvsh, "MTimeHumidity",     "FillCamHum");
    fillvs1.SetNameTab("ObsTime");
    fillvs2.SetNameTab("OnTime");
    fillvst1.SetNameTab("Rate");
    fillvst2.SetNameTab("RateZd");
    fillvst.SetNameTab("CamTemp");
    fillvsh.SetNameTab("CamHum");

    hvs1.SetWeight("MReportRates.fElapsedTime");
    hvs2.SetWeight("MReportRates.fElapsedOnTime");

    hvst1.SetMinimum(0);
    hvst2.SetDrawOption("colz");

    MH3 hrate("MRawRunHeader.GetFileID"/*, "MRawEvtHeader.GetTriggerID"*/);
    hrate.SetWeight("1./TMath::Max(MRawRunHeader.GetRunLength,1)");
    hrate.SetName("RateVsRun");
    hrate.SetTitle("Event rate after quality cuts;File Id;Event Rate [Hz];");
    hrate.InitLabels(MH3::kLabelsX);
    hrate.GetHist().SetMinimum(0);

    MFillH fillr(&hrate, "", "FillRateVsRun");
    tlist.AddToListAfter(&fillr, &contq, "Events");

    tlist.Replace(&fill1b);

    tlist2.Replace(&fill2b);
    tlist2.Replace(&fill3b);
    tlist2.Replace(&fill3d);
    tlist2.Replace(&fill4b);
    tlist2.Replace(&fill5b);
    tlist2.Replace(&fill6b);
    tlist2.Replace(&fill7b);
    
    std::cout << "test vor falpha2";

    /*if (!ismc)
        tlist2.Replace(&falpha2);
    */

   // tlist2.Replace(&fenergy2);
    if (!ismc) // added from MC ganymed
    {
        tlist.AddToList(&fillvs1,  "Rates");
        tlist.AddToList(&fillvs2,  "Rates");
        tlist.AddToList(&fillvst1, "Rates");
        tlist.AddToList(&fillvst2, "Rates");
        tlist.AddToList(&write2,   "Rates");
        tlist.AddToList(&fillvst,  "Temperatures");
        tlist.AddToList(&fillvsh,  "Humidity");
        tlist.AddToList(&fillvst0, "Drive");
        tlist.AddToList(&fillvstM, "Drive");
    }

    // by setting it here it is distributed to all consecutive tasks
    tlist.SetAccelerator(MTask::kAccDontReset|MTask::kAccDontTime);

    par.SetVal(1);
  //  weight.SetVal(1); //added from MC ganymed

    // Execute first analysis

    if (!evtloop.Eventloop())
        return 2;


    if (!evtloop.GetDisplay())
        return 2;


    TObjArray cont;
    //cont.Add(const_cast<TEnv*>(GetEnv()));
    //cont.Add(const_cast<MSequence*>(&fSequence));

    //TNamed cmdline("CommandLine", fCommandLine.Data());
    //cont.Add(&cmdline);

    if (d)
    {
        TString title = "--  Analysis #";
        title += gSystem->BaseName(outfile);
        title += "  --";
        d->SetTitle(title, kFALSE);

        cont.Add(d);
        TString path;
        path += fname1;

        d->Write(); //d->SaveAs(path);
    }

//    if (!WriteContainer(cont, fname1, "RECREATE"))
//        return;

    return 0;
}

int ganymed(const char *dataset, const char *outfile, Float_t ra, Float_t dec, Bool_t ismc=false)
{
    MDirIter files;
    if (files.ReadList(dataset)<=0)
        return 2;
    files.Sort();
    return process(files, outfile, ra, dec, ismc);
}


int ganymed(Float_t ra, Float_t dec, const char *starfile, const char *outfile, Bool_t ismc=false)
{
    MDirIter files;
    files.AddFile(starfile);
    return process(files, outfile, ra, dec, ismc);
}

