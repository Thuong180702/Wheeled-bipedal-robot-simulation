\@ifundefined{ExplSyntaxOn}{\RequirePackage{expl3}}{}
\ProvidesExplClass{jcd}{2025/09/24}{v1.3}{jcd}
\keys_define:nn { jcd/options }
  {
    unknown .code:n = \PassOptionsToClass{#1}{article}
  }
\PassOptionsToClass{twoside}{article}
\cs_if_exist:NTF \ProcessKeyOptions
  { \ProcessKeyOptions[jcd/options] }
  { \RequirePackage{l3keys2e} \ProcessKeysOptions{jcd/options} }
\ExplSyntaxOff
\LoadClass{article}

\renewcommand\normalsize{%
   \@setfontsize\normalsize{10.56bp}{12.6bp}%
   \abovedisplayskip 4\p@ \@plus2\p@ \@minus2\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@
   \belowdisplayskip \abovedisplayskip
   \let\@listi\@listI}
\normalsize
\DeclareRobustCommand\small{%
   \@setfontsize\small{9bp}{11bp}%
   \abovedisplayskip 3\p@ \@plus3\p@ \@minus2\p@
   \abovedisplayshortskip \z@ \@plus2\p@
   \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 4\p@ \@plus2\p@ \@minus2\p@
               \parsep 2\p@ \@plus\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}
\DeclareRobustCommand\footnotesize{%
   \@setfontsize\footnotesize{8bp}{9.6bp}%
   \abovedisplayskip 3\p@ \@plus2\p@ \@minus2\p@
   \abovedisplayshortskip \z@ \@plus\p@
   \belowdisplayshortskip 3\p@ \@plus\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 3\p@ \@plus\p@ \@minus\p@
               \parsep 2\p@ \@plus\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}
\DeclareRobustCommand\scriptsize{\@setfontsize\scriptsize{7.5bp}{9bp}}
\DeclareRobustCommand\tiny{\@setfontsize\tiny{7bp}{8.4bp}}
\DeclareRobustCommand\large{\@setfontsize\large{11bp}{13.2bp}}
\DeclareRobustCommand\Large{\@setfontsize\Large{14bp}{16.8bp}}
\DeclareRobustCommand\LARGE{\@setfontsize\LARGE\@xviipt{22}}
\DeclareRobustCommand\huge{\@setfontsize\huge\@xxpt{25}}
\DeclareRobustCommand\Huge{\@setfontsize\Huge\@xxvpt{30}}
\linespread{1.2}\selectfont

\RequirePackage{microtype}
\setlength\xspaceskip{0.333em plus 0.266em minus 0.166em}
\setlength\spaceskip{0.333em plus 0.266em minus 0.166em}
\def\@normalspace{\xspaceskip\z@skip \spaceskip\z@skip}
% \RequirePackage[usegeometry,paper=29.39cm:22.38cm]{typearea}
\RequirePackage{geometry}
\geometry{a4paper, hmargin=19.7mm, top=20mm, textheight=715pt, nomarginpar, headsep=2mm, includefoot}
\def\@resettonormalpage{\thispagestyle{firstpage}}
\RequirePackage{atbegshi}
\AtBeginShipout{%
  \ifnum\c@page=1 \@changefirstpagesize
  \else\ifnum\c@page=2 \@changesecondpagesize
    \else\@changeafterpagesize
    \fi
  \fi}
\def\@changefirstpagesize{}
\def\@changesecondpagesize{\global\textheight 737\p@}
\def\@changeafterpagesize{}

\RequirePackage{amsmath, amsfonts, amsthm}
\counterwithin{equation}{section}
\newtheoremstyle{plain}
  {0mm}
  {0mm}
  {\itshape}
  {}
  {\bfseries}
  {:}
  {.5em}
  {}
\newtheoremstyle{definition}
  {0mm}
  {0mm}
  {\rmfamily}
  {}
  {\bfseries}
  {:}
  {.5em}
  {}
\AtBeginDocument{
  \@ifundefined{blacksquare}{%
    \renewcommand{\qedsymbol}{\ensuremath{\mdlgblksquare}}}{%
    \renewcommand{\qedsymbol}{\ensuremath{\blacksquare}}}%
}
\renewenvironment{proof}[1][\proofname]{\par
  \pushQED{\qed}%
  \normalfont \topsep0\p@\@plus6\p@\relax
    \parsep0\p@ \partopsep0\p@
  \trivlist
  \item\relax{\bfseries\itshape#1\@addpunct{:}}\hspace\labelsep\ignorespaces
}{%
  \popQED\endtrivlist\@endpefalse
}

\RequirePackage{etoolbox}
\RequirePackage{graphicx}
\graphicspath{{figures}{images}{pictures}{logos}}
\RequirePackage{xcolor}
\definecolor{main}{HTML}{ea4e00}
\definecolor{sub}{HTML}{ffede8}
\definecolor{ref}{HTML}{000084}

\RequirePackage{colortbl}
\RequirePackage{makecell}
\arrayrulecolor{main}

\RequirePackage{caption,subcaption}
\DeclareCaptionFont{capfam}{\rmfamily}
\DeclareCaptionFont{capsize}{\fontsize{9.5bp}{11.4bp}\selectfont}
\DeclareCaptionFont{subcapsize}{\small}
\DeclareCaptionFont{colored}{\color{main}}
% \DeclareCaptionFormat{fullwidth}{\parbox[t]{\linewidth}{%
%   \@hangfrom{#1#2}%
%   \advance\caption@parindent\hangindent\relax
%   \advance\caption@hangindent\hangindent\relax
%   \caption@@par#3\par}}
\def\captionwidth{\linewidth}
\DeclareCaptionFormat{fullwidth}{\parbox[t]{\captionwidth}{#1#2#3}}
\captionsetup[table]{
  format   = fullwidth,
  font     = {capfam,capsize},
  labelfont={capfam,bf,capsize,colored},
  labelsep = period,
  skip     = 1mm,
  position = top,
}
\captionsetup[figure]{
  format   = fullwidth,
  font     = {capfam,capsize},
  labelfont={capfam,bf,capsize,colored},
  labelsep = period,
  skip     = 1mm,
  position = bottom,
}
\captionsetup[subtable]{
  format   = plain,
  font     = {capfam,subcapsize},
  labelfont= {capfam,subcapsize},
  labelsep =none,
}
\captionsetup[subfigure]{
  format   = plain,
  font     = {capfam,subcapsize},
  labelfont= {capfam,subcapsize},
  labelsep =space,
}
\setlength\dblfloatsep{7pt plus 3pt minus 3pt}
\setlength\dbltextfloatsep{7pt plus 3pt minus 3pt}
\setlength\intextsep{7pt plus 3pt minus 3pt}


\RequirePackage{enumitem}
\setlist[itemize,1]{nosep, leftmargin=17pt, labelwidth=17pt, labelsep=0pt, itemindent=0pt, align=left}
\setlist[enumerate,1]{nosep, leftmargin=17pt, labelwidth=17pt, labelsep=0pt, itemindent=0pt, align=left, label={(\arabic*)}}

\RequirePackage[noindentafter]{titlesec}
\def\thesection{\arabic{section}.}
\def\theequation{\arabic{equation}}
\titleformat{\section}[hang]
  {\@normalspace\sffamily\large\bfseries\color{main}\raggedright}
  {\thetitle}
  {.5em}
  {}
  {}
\titlespacing{\section}
  {10pt}
  {3mm plus 1mm minus 1mm}
  {3mm plus 1mm minus 1mm}
\def\thesubsection{\arabic{section}.\arabic{subsection}.}
\titleformat{\subsection}[hang]
  {\@normalspace\sffamily\normalsize\bfseries\itshape\color{main}}
  {\thetitle}
  {.5em}
  {}
  {}
\titlespacing{\subsection}
  {0pt}
  {3mm plus .5mm minus .5mm}
  {3mm plus .5mm minus .5mm}
\titleformat{\subsubsection}[hang]
  {\@normalspace\sffamily\normalsize\bfseries\itshape\color{main}}
  {\thetitle}
  {.5em}
  {}
  {}
\titlespacing{\subsubsection}
  {0pt}
  {1mm plus .5mm minus .5mm}
  {1mm plus .5mm minus .5mm}
\def\thesubsubsection{\arabic{section}.\arabic{subsection}.\arabic{subsubsection}}

\newcommand\backmatter{%
  \def\thesection{}%
  \titleformat{\section}[block]
    {\sffamily\large\bfseries\color{main}\raggedright}
    {}
    {0pt}
    {}
    {}%
  \titlespacing{\section}
    {0pt}
    {3mm plus .5mm minus .5mm}
    {2mm plus .5mm minus .5mm}
  \small
}

\setlength{\columnsep}{6mm}
\setlength{\parindent}{0bp}

\newcommand\maketitlepage[1][]{%
  \gdef\maketitlepage{}%
  {#1}%
  \@resettonormalpage
}

\def\@attention{}
\NewDocumentEnvironment{attention}{+b}{\gdef\@attention{#1}}{}

\RequirePackage{balance}
\newbox\@abstract@box
\renewenvironment{abstract}
  {\begin{lrbox}{\@abstract@box}\minipage[t]{115mm}
    \@normalspace\sffamily\linespread{1.3}\selectfont
    \vspace{3mm}%
    \mbox{\bfseries\scriptsize\color{main}ABSTRACT}%
    \par\fontsize{9bp}{9.8bp}\selectfont}
  {\par\vspace{3mm}\endminipage\end{lrbox}%
    \twocolumn[{\vspace{12mm}\@maketitle@abstract[\@attention]}]}
\newcommand\@maketitle@abstract[1][]{%
  {\Large\@normalspace\sffamily\bfseries\raggedright\color{main}\@title\par}%
  \vspace{1mm}%
  {\normalsize\@normalspace\sffamily\@printauthors\par}%
  \vspace{1.5mm}%
  {\small\@normalspace\sffamily\@printaffiliations\par}%
  \vspace{7mm}%
  {#1}\par
  \hbox to\textwidth{\fboxrule\z@ \fboxsep4mm \fcolorbox{sub}{sub}{\raisebox{\height}{\box\@abstract@box}}\hskip4mm
  \parbox[t]{31mm}{\vspace*{1mm}\@normalspace\sffamily\raggedright\@history@keywords}\hfil}%
  \vspace{7.5mm}%
}
\def\@history@keywords{%
  \fontsize{8bp}{9.5bp}\selectfont
  {\bfseries\scriptsize\color{main}ARTICLE HISTORY}\\%
  \@history
  \par
  \vspace*{3mm}%
  {\bfseries\scriptsize\color{main}KEYWORDS}\\%
  \@keywords
}
% \newcommand\printkeywords[1][\@keywords]{{\bfseries\scriptsize\color{main}KEYWORDS}\\#1}

\RequirePackage{array,tabularx,booktabs}
% \setlength{\tabcolsep}{0pt}

% -- biography
\newlength\beforebioskip \let\biobeforeskip\beforebioskip
\setlength{\beforebioskip}{3.5mm}
\newlength\afterbioskip \let\biobeafterskip\afterbioskip
\newlength{\biofigurewidth}
\setlength{\biofigurewidth}{26mm}

\RequirePackage{needspace}
\newdimen\@jcdtmpdimenA
\newdimen\@jcdtmpdimenB
\newcount\@jcdtmpcountA
\def\@jcdBIOphotowidth{\biofigurewidth}
\def\@jcdBIOphotodepth{30mm}
\def\@jcdBIOhangwidth{28mm}
\def\@jcdBIOhangdepth{30mm}
\def\@jcdBIOskipN{0\baselineskip}
\newenvironment{jcdbiography}[3][width=26mm]
  {\unitlength 1in\parskip=0pt\par \interlinepenalty500
    \@jcdtmpdimenA=\@jcdBIOhangdepth%
    \advance\@jcdtmpdimenA by \@jcdBIOskipN%
    \advance\@jcdtmpdimenA by 1\baselineskip%
    \nointerlineskip
    \needspace{.8\@jcdtmpdimenA}%
    \vskip\beforebioskip
    \def\@jcdtempbiographybox
      {\mbox{\begin{minipage}[b][\@jcdBIOphotodepth][c]{\@jcdBIOphotowidth}%
        \centering \includegraphics[#1]{#3}\end{minipage}}}%
    \let\@jcdBIOORGparCMD=\par 
    \edef\par{\hfil\break\indent}%  
    \settoheight{\@jcdtmpdimenA}{\@jcdtempbiographybox}% 
    \@jcdtmpdimenB=\@jcdBIOhangdepth 
    \@jcdtmpcountA=\@jcdtmpdimenB 
    \divide\@jcdtmpcountA by \baselineskip 
    \advance\@jcdtmpcountA by 1 
    \hangindent\@jcdBIOhangwidth
    \hangafter-\@jcdtmpcountA 
    \settoheight{\@jcdtmpdimenB}{\mbox{T}}% 
    \noindent\makebox[0pt][l]{\hspace{-\@jcdBIOhangwidth}\raisebox{\@jcdtmpdimenB}[0pt][0pt]{\raisebox{-\@jcdBIOphotodepth}[0pt][0pt]{\@jcdtempbiographybox}}}%
    \noindent{\itshape\bfseries\color{main}#2}\space\ignorespaces}
  {\relax\let\par=\@jcdBIOORGparCMD\par
    \ifnum \prevgraf <\@jcdtmpcountA\relax 
      \advance\@jcdtmpcountA by -\prevgraf 
      \advance\@jcdtmpcountA by -1\relax 
      \@jcdtmpdimenA=\baselineskip 
      \multiply\@jcdtmpdimenA by \@jcdtmpcountA
      \noindent\rule{0pt}{\@jcdtmpdimenA}%  
    \fi \par\normalfont}
\newenvironment{contributor}{\jcdbiography}{\endjcdbiography}

\RequirePackage{apacite}
\def\doiprefix{}
\renewcommand{\APACrefbtitle}[2]{\Bem{#1}}
\bibliographystyle{apacite}
\RequirePackage[authoryear]{natbib}

\RequirePackage{xurl}
\def\UrlBreaks{%
  \do\/%
  \do\a\do\b\do\c\do\d\do\e\do\f\do\g\do\h\do\i\do\j\do\k\do\l%
     \do\m\do\n\do\o\do\p\do\q\do\r\do\s\do\t\do\u\do\v\do\w\do\x\do\y\do\z%
  \do\A\do\B\do\C\do\D\do\E\do\F\do\G\do\H\do\I\do\J\do\K\do\L%
     \do\M\do\N\do\O\do\P\do\Q\do\R\do\S\do\T\do\U\do\V\do\W\do\X\do\Y\do\Z%
  \do0\do1\do2\do3\do4\do5\do6\do7\do8\do9\do=\do/\do.\do:%
  \do\*\do\-\do\~\do\'\do\"\do\-}

\newcommand\@jcdul[1]{{\setbox\@tempboxa\hbox{#1}%
  \hbox{\vrule\@width\wd\@tempboxa
    \@depth\dimexpr.2ex+.2pt\relax\@height-.2ex\kern-\wd\@tempboxa\box\@tempboxa}}}
\DeclareRobustCommand*\ULurl[1]{\href{#1}{\@jcdul{#1}}}
\DeclareRobustCommand*\ULnolinkurl{\@jcdul}

\setlength{\bibsep}{14pt}
\newcommand\@setzerosep{\parsep4\p@ \topsep\z@ \partopsep\z@ \itemsep\z@ \addtolength\leftmargin{6\p@}}
\renewenvironment{thebibliography}[1]{%
 \bibsection
 \parindent\z@
 \bibpreamble
 \bibfont
 \list{\@biblabel{\the\c@NAT@ctr}}{\@bibsetup{#1}\global\c@NAT@ctr\z@\@setzerosep}%
 \ifNAT@openbib
   \renewcommand\newblock{\par}%
 \else
   \renewcommand\newblock{\hskip .11em \@plus.33em \@minus.07em}%
 \fi
 \sloppy\clubpenalty4000\widowpenalty4000
 \sfcode`\.\@m
 \let\NAT@bibitem@first@sw\@firstoftwo
    \let\citeN\cite \let\shortcite\cite
    \let\citeasnoun\cite
}{%
 \bibitem@fin
 \bibpostamble
 \def\@noitemerr{%
  \PackageWarning{natbib}{Empty `thebibliography' environment}%
 }%
 \endlist
 \bibcleanup
}%
\renewcommand{\bibsection}{}
\urlstyle{rm}
\DeclareRobustCommand\@doi[1]{\href{https://doi.org/#1}{\nolinkurl{https://doi.org/#1}}}
\DeclareRobustCommand{\ULdoi}[1]{\href{https//doi.org/#1}{\ULnolinkurl{https://doi.org/#1}}}
\DeclareRobustCommand\@email[1]{\href{mailto:#1}{\nolinkurl{#1}}}
\DeclareRobustCommand{\ULemail}[1]{\href{mailto:#1}{\ULnolinkurl{#1}}}
\let\doi\@doi
\AtBeginDocument{%
  \let\JCD@ori@hyper@natlinkstart\hyper@natlinkstart
  \renewcommand\hyper@natlinkstart[1]{\Hy@raisedlink{\hyper@anchorstart{citeback.#1}}\hyper@anchorend\JCD@ori@hyper@natlinkstart{#1}}
}

\ExplSyntaxOn

\cs_new_protected:Npn \jcdsetup { \keys_set:nn { jcd } }
\cs_set_eq:NN \documentsetup \jcdsetup
\keys_define:nn { jcd }
  {
    title .tl_set:N = \@title ,
    history .tl_set:N = \@history ,
    keywords .tl_set:N = \@keywords ,
    authors .code:n = \@setauthors{#1} ,
    affiliations .code:n = \@setaffiliations{#1} ,
    short-author .tl_set:N = \@shortauthor ,
    journal .tl_set:N = \@journal ,
    journal .initial:n = {Journal~of~Control~and~Decision} ,
    start-page .tl_set:N = \@startpage ,
    end-page   .tl_set:N = \@endpage ,
    start-page .initial:n = 1 ,
    end-page   .initial:n = 1 ,
    doi .tl_set:N = \@docdoi ,
    year .tl_set:N = \@docyear ,
    volume .tl_set:N = \@docvolume ,
    number .tl_set:N = \@docnumber ,
    cite .tl_set:N = \@doccite ,
    cite-me .tl_set:N = \@doccite ,
    contact-to .tl_set:N = \@contactto ,
    contact-email .tl_set:N = \@contactemail ,
    contact-address .tl_set:N = \@contactaddress ,
    copyright .tl_set:N = \@doccopyright ,
    homepage .tl_set:N = \@homepage ,
    homepage .initial:n = {www.tandfonline.com/journals/tjcd20} ,
    issnprint .tl_set:N = \@issnprint ,
    online-date .tl_set:N = \@onlinedate ,
    views .tl_set:N = \@docviews ,
    notes .tl_set:N = \@notes ,
  }
\cs_new:Npn \@setauthors
  { \keyval_parse:NNn \@gobble \@setauthors@ }
\cs_new:Npn \@setaffiliations
  { \keyval_parse:NNn \@gobble \@setaffiliations@ }
\seq_new:N \g__jcd_authors_seq
\def\@setauthors@#1#2{
  \seq_gput_right:Nn \g__jcd_authors_seq {#1}
  \@namedef{author-#1}{#2}
}
\cs_new:Npn \@printauthors
  {
    \int_case:nnF { \seq_count:N \g__jcd_authors_seq }
      {
        0 { }
        1 { \exp_args:Ne\@oneauthor{\seq_item:Nn \g__jcd_authors_seq{1}} }
        2 {
          \exp_args:Ne\@oneauthor{\seq_item:Nn \g__jcd_authors_seq{1}},~and~
          \exp_args:Ne\@oneauthor{\seq_item:Nn \g__jcd_authors_seq{2}}
        }
      }
      {
        \int_step_inline:nn { \seq_count:N \g__jcd_authors_seq-1 }
          {
            \exp_args:Ne\@oneauthor{\seq_item:Nn \g__jcd_authors_seq{##1}},\space
          }
          \space and\space
          \exp_args:Ne\@oneauthor{\seq_item:Nn \g__jcd_authors_seq{-1}}
      }
  }
\cs_new:Npn \@titlepageauthors
  {
    \seq_use:Nnnn \g__jcd_authors_seq
      { ,~ } { ,~ } { ~\&~ }
  }
\def\@oneauthor#1{
  #1 \textsuperscript{\scriptsize\@nameuse{author-#1}}
}
\prop_new:N \g__jcd_affiliations_prop
\def\@setaffiliations@#1#2{
  \prop_gput:Nnn \g__jcd_affiliations_prop {#1} {#2}
}
\cs_new:Npn \@printaffiliations
  {
    \seq_clear:N \l__jcd_tmpa_seq
    \prop_map_inline:Nn \g__jcd_affiliations_prop
      {
        \seq_put_right:Nn \l__jcd_tmpa_seq {##1}
      }
    \seq_pop_right:NN \l__jcd_tmpa_seq \l__jcd_tmpl_tl
    \seq_map_inline:Nn \l__jcd_tmpa_seq
      {
        \@oneaffiliation{##1};\space
      }
    \quark_if_no_value:NF \l__jcd_tmpl_tl
      {\exp_args:No\@oneaffiliation{\l__jcd_tmpl_tl}}
  }
\def\@oneaffiliation#1{
  \textsuperscript{\tiny#1}\prop_item:Nn \g__jcd_affiliations_prop{#1}
}
\ExplSyntaxOff

\RequirePackage{calc}
\def\@headlogo{\raisebox{-.45em}{\includegraphics[width=1.6em]{small-logo}}}
% \def\@jcdthepage{\number\numexpr\@startpage+\c@page-1-1\relax}
\def\@jcdthepage{\number\c@page\relax}
\def\ps@main{%
  % \def\@oddhead{\normalfont\normalcolor
  %   \scriptsize\@normalspace\sffamily\@jcdthepage\hspace{1em}%
  %   \@headlogo\hspace{1em}%
  %   \MakeUppercase\@shortauthor\hfil}%
  \def\@oddhead{\normalfont\normalcolor
    \scriptsize\@normalspace\sffamily\hfil\MakeUppercase\@journal\hspace{7mm}%
    \@headlogo\hspace{7mm}%
    \@jcdthepage}%
  \def\@evenhead{\normalfont\normalcolor
    \scriptsize\@jcdthepage\hfil}%
  \def\@oddfoot{}%
  \def\@evenfoot{}%
}
\pagestyle{main}
\definecolor{blueg}{HTML}{162575}
\def\ps@titlepage{%
  \def\@oddhead{\hbox to\textwidth{\hss\begin{minipage}[b]{\dimexpr\paperwidth-15mm}
    \hypersetup{hidelinks}%
    \hbox to\linewidth{\sffamily\bfseries\@normalspace
      \vbox to44mm{\hsize33mm\includegraphics[width=33mm,height=44mm]{journal-cover}}\hskip4.5mm
      \vbox to 44mm{\hsize145mm\vfil
        {\fontsize{16bp}{16bp}\selectfont\@journal}\par
        \vspace*{25mm}
        {\fontsize{8bp}{8bp}\selectfont
        \@issnprint\space
        Journal Homepage: \ULurl{\@homepage}}\par
        \hrule\@height\z@\@depth\z@}\hfil
      \vbox to44mm{\llap{\includegraphics[width=26mm,height=6.mm]{group-blue}}\vss}}
  \vspace*{1.5mm}
  \color{blueg}
  \hrule\@width\linewidth\@height.66mm\@depth\z@
  \end{minipage}\hss}}%
  \let\@evenhead\@oddhead
  \def\@oddfoot{\begin{minipage}{\textwidth}\centering
    \hypersetup{hidelinks}\fontsize{9.7bp}{11.5bp}\sffamily\def\UrlFont{\sffamily}
    Full Terms \& Conditions of access and use can be found at\\
    \url{https://www.tandfonline.com/action/journalInformation?journalCode=tjcd20}
    \vspace*{18mm}
  \end{minipage}}%
  \let\@evenfoot\@oddfoot
}
\def\@makegrouplogo{\href{http://www.tandfonline.com}{\XeTeXLinkBox{\includegraphics[width=36mm,height=9mm]{group-logo}}}}
\def\ps@firstpage{%
  \def\@oddhead{\raisebox{-11mm}[0mm][9mm]{\vbox{\hsize\textwidth
    \hbox to\textwidth{%
      \hypersetup{hidelinks}%
      \parbox[c]{\textwidth}{\raggedright
        \scriptsize\sffamily\@normalspace
        \MakeUppercase\@journal
      }}%
    \vskip1.8mm
    {\color{main}\hrule width\textwidth depth\z@ height0.4mm}%
    \vskip.8mm
    \hbox to\textwidth{\hfil
      \includegraphics[width=21mm,height=4mm]{check-updates}}}}}%
  \let\@evenhead\@oddhead
  \def\@oddfoot{\raisebox{-6.5mm}{\vbox{\hsize\textwidth
    % \hypersetup{hidelinks}%
    % {\color{main}\hrule width\textwidth depth\z@ height0.4mm}%
    % \vskip1.4mm
    \fontsize{8bp}{9bp}\sffamily\@normalspace
    {\color{main}\bfseries CONTACT}~\@contactto
    \space\raisebox{-1.2bp}{\includegraphics[width=9bp]{email}}\space\href{mailto:\@contactemail}{\@contactemail}
    \space\raisebox{-1.2bp}{\includegraphics[width=9bp]{card}}\space\@contactaddress
    \ifx\@notes\@empty\else
      \space\raisebox{-1.2bp}{\includegraphics[width=9bp]{notes}}\space\@notes
    \fi
  }}}%
  \let\@evenfoot\@oddfoot
}

\ExplSyntaxOn
