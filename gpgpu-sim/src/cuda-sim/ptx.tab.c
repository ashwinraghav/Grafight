
/* A Bison parser, made by GNU Bison 2.4.1.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C
   
      Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.
   
   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.4.1"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Using locations.  */
#define YYLSP_NEEDED 0

/* Substitute the variable and function names.  */
#define yyparse         ptx_parse
#define yylex           ptx_lex
#define yyerror         ptx_error
#define yylval          ptx_lval
#define yychar          ptx_char
#define yydebug         ptx_debug
#define yynerrs         ptx_nerrs


/* Copy the first part of user declarations.  */


/* Line 189 of yacc.c  */
#line 81 "ptx.tab.c"

/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     STRING = 258,
     OPCODE = 259,
     ALIGN_DIRECTIVE = 260,
     BYTE_DIRECTIVE = 261,
     CONST_DIRECTIVE = 262,
     ENTRY_DIRECTIVE = 263,
     EXTERN_DIRECTIVE = 264,
     FILE_DIRECTIVE = 265,
     FUNC_DIRECTIVE = 266,
     GLOBAL_DIRECTIVE = 267,
     LOCAL_DIRECTIVE = 268,
     LOC_DIRECTIVE = 269,
     PARAM_DIRECTIVE = 270,
     REG_DIRECTIVE = 271,
     SECTION_DIRECTIVE = 272,
     SHARED_DIRECTIVE = 273,
     SREG_DIRECTIVE = 274,
     STRUCT_DIRECTIVE = 275,
     SURF_DIRECTIVE = 276,
     TARGET_DIRECTIVE = 277,
     TEX_DIRECTIVE = 278,
     UNION_DIRECTIVE = 279,
     VERSION_DIRECTIVE = 280,
     VISIBLE_DIRECTIVE = 281,
     MAXNTID_DIRECTIVE = 282,
     IDENTIFIER = 283,
     INT_OPERAND = 284,
     FLOAT_OPERAND = 285,
     DOUBLE_OPERAND = 286,
     S8_TYPE = 287,
     S16_TYPE = 288,
     S32_TYPE = 289,
     S64_TYPE = 290,
     U8_TYPE = 291,
     U16_TYPE = 292,
     U32_TYPE = 293,
     U64_TYPE = 294,
     F16_TYPE = 295,
     F32_TYPE = 296,
     F64_TYPE = 297,
     B8_TYPE = 298,
     B16_TYPE = 299,
     B32_TYPE = 300,
     B64_TYPE = 301,
     PRED_TYPE = 302,
     V2_TYPE = 303,
     V3_TYPE = 304,
     V4_TYPE = 305,
     COMMA = 306,
     PRED = 307,
     EQ_OPTION = 308,
     NE_OPTION = 309,
     LT_OPTION = 310,
     LE_OPTION = 311,
     GT_OPTION = 312,
     GE_OPTION = 313,
     LO_OPTION = 314,
     LS_OPTION = 315,
     HI_OPTION = 316,
     HS_OPTION = 317,
     EQU_OPTION = 318,
     NEU_OPTION = 319,
     LTU_OPTION = 320,
     LEU_OPTION = 321,
     GTU_OPTION = 322,
     GEU_OPTION = 323,
     NUM_OPTION = 324,
     NAN_OPTION = 325,
     LEFT_SQUARE_BRACKET = 326,
     RIGHT_SQUARE_BRACKET = 327,
     WIDE_OPTION = 328,
     SPECIAL_REGISTER = 329,
     PLUS = 330,
     COLON = 331,
     SEMI_COLON = 332,
     EXCLAMATION = 333,
     RIGHT_BRACE = 334,
     LEFT_BRACE = 335,
     EQUALS = 336,
     PERIOD = 337,
     DIMENSION_MODIFIER = 338,
     RN_OPTION = 339,
     RZ_OPTION = 340,
     RM_OPTION = 341,
     RP_OPTION = 342,
     RNI_OPTION = 343,
     RZI_OPTION = 344,
     RMI_OPTION = 345,
     RPI_OPTION = 346,
     UNI_OPTION = 347,
     GEOM_MODIFIER_1D = 348,
     GEOM_MODIFIER_2D = 349,
     GEOM_MODIFIER_3D = 350,
     SAT_OPTION = 351,
     FTZ_OPTION = 352,
     ATOMIC_AND = 353,
     ATOMIC_OR = 354,
     ATOMIC_XOR = 355,
     ATOMIC_CAS = 356,
     ATOMIC_EXCH = 357,
     ATOMIC_ADD = 358,
     ATOMIC_INC = 359,
     ATOMIC_DEC = 360,
     ATOMIC_MIN = 361,
     ATOMIC_MAX = 362,
     LEFT_ANGLE_BRACKET = 363,
     RIGHT_ANGLE_BRACKET = 364,
     LEFT_PAREN = 365,
     RIGHT_PAREN = 366,
     APPROX_OPTION = 367,
     FULL_OPTION = 368,
     ANY_OPTION = 369,
     ALL_OPTION = 370,
     GLOBAL_OPTION = 371,
     CTA_OPTION = 372
   };
#endif



#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 214 of yacc.c  */
#line 64 "ptx.y"

  double double_value;
  float  float_value;
  int    int_value;
  char * string_value;
  void * ptr_value;



/* Line 214 of yacc.c  */
#line 244 "ptx.tab.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */

/* Line 264 of yacc.c  */
#line 190 "ptx.y"

  	#include "ptx_ir.h"
	#include <stdlib.h>
	#include <string.h>
	#include <math.h>
	void syntax_not_implemented();
	extern int g_func_decl;
	int ptx_lex(void);
	int ptx_error(const char *);


/* Line 264 of yacc.c  */
#line 268 "ptx.tab.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int yyi)
#else
static int
YYID (yyi)
    int yyi;
#endif
{
  return yyi;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)				\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack_alloc, Stack, yysize);			\
	Stack = &yyptr->Stack_alloc;					\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  2
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   490

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  118
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  53
/* YYNRULES -- Number of rules.  */
#define YYNRULES  181
/* YYNRULES -- Number of states.  */
#define YYNSTATES  266

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   372

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   108,   109,   110,   111,   112,   113,   114,
     115,   116,   117
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     4,     7,    10,    13,    14,    20,    21,
      28,    35,    36,    43,    44,    48,    50,    51,    57,    59,
      61,    63,    65,    69,    70,    75,    76,    81,    83,    85,
      88,    91,    94,    97,   102,   105,   109,   114,   117,   122,
     127,   129,   131,   135,   137,   142,   146,   151,   153,   156,
     158,   160,   162,   164,   167,   169,   171,   173,   175,   177,
     179,   181,   183,   185,   187,   189,   192,   194,   196,   198,
     200,   202,   204,   206,   208,   210,   212,   214,   216,   218,
     220,   222,   224,   226,   228,   230,   234,   238,   240,   244,
     247,   250,   254,   255,   267,   274,   277,   279,   280,   284,
     286,   289,   293,   295,   298,   300,   302,   304,   306,   308,
     310,   312,   314,   316,   318,   320,   322,   324,   326,   328,
     330,   332,   334,   336,   338,   340,   342,   344,   346,   348,
     350,   352,   354,   356,   358,   360,   362,   364,   366,   368,
     370,   372,   374,   376,   378,   380,   382,   384,   386,   388,
     390,   392,   394,   396,   398,   400,   402,   404,   406,   408,
     410,   412,   416,   418,   421,   423,   425,   427,   429,   431,
     435,   441,   449,   459,   473,   476,   478,   482,   484,   486,
     488,   490
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     119,     0,    -1,    -1,   119,   135,    -1,   119,   120,    -1,
     119,   124,    -1,    -1,   124,   121,    80,   134,    79,    -1,
      -1,   124,   122,   123,    80,   134,    79,    -1,    27,    29,
      51,    29,    51,    29,    -1,    -1,   129,   110,   125,   131,
     111,   127,    -1,    -1,   129,   126,   127,    -1,   129,    -1,
      -1,    28,   128,   110,   130,   111,    -1,    28,    -1,     8,
      -1,    11,    -1,   131,    -1,   130,    51,   131,    -1,    -1,
      15,   132,   137,   139,    -1,    -1,    16,   133,   137,   139,
      -1,   135,    -1,   150,    -1,   134,   135,    -1,   134,   150,
      -1,   136,    77,    -1,    25,    31,    -1,    22,    28,    51,
      28,    -1,    22,    28,    -1,    10,    29,     3,    -1,    14,
      29,    29,    29,    -1,   137,   138,    -1,   137,   139,    81,
     148,    -1,   137,   139,    81,   169,    -1,   140,    -1,   139,
      -1,   138,    51,   139,    -1,    28,    -1,    28,   108,    29,
     109,    -1,    28,    71,    72,    -1,    28,    71,    29,    72,
      -1,   141,    -1,   140,   141,    -1,   143,    -1,   145,    -1,
     142,    -1,     9,    -1,     5,    29,    -1,    16,    -1,    19,
      -1,   144,    -1,     7,    -1,    12,    -1,    13,    -1,    15,
      -1,    18,    -1,    21,    -1,    23,    -1,   147,    -1,   146,
     147,    -1,    48,    -1,    49,    -1,    50,    -1,    32,    -1,
      33,    -1,    34,    -1,    35,    -1,    36,    -1,    37,    -1,
      38,    -1,    39,    -1,    40,    -1,    41,    -1,    42,    -1,
      43,    -1,    44,    -1,    45,    -1,    46,    -1,    47,    -1,
      80,   149,    79,    -1,    80,   148,    79,    -1,   169,    -1,
     169,    51,   149,    -1,   151,    77,    -1,    28,    76,    -1,
     155,   151,    77,    -1,    -1,   153,   110,   164,   111,   152,
      51,   164,    51,   110,   163,   111,    -1,   153,   164,    51,
     110,   163,   111,    -1,   153,   163,    -1,   153,    -1,    -1,
       4,   154,   156,    -1,     4,    -1,    52,    28,    -1,    52,
      78,    28,    -1,   157,    -1,   157,   156,    -1,   145,    -1,
     162,    -1,   144,    -1,   159,    -1,    92,    -1,    73,    -1,
     114,    -1,   115,    -1,   116,    -1,   117,    -1,    93,    -1,
      94,    -1,    95,    -1,    96,    -1,    97,    -1,   112,    -1,
     113,    -1,   158,    -1,    98,    -1,    99,    -1,   100,    -1,
     101,    -1,   102,    -1,   103,    -1,   104,    -1,   105,    -1,
     106,    -1,   107,    -1,   160,    -1,   161,    -1,    84,    -1,
      85,    -1,    86,    -1,    87,    -1,    88,    -1,    89,    -1,
      90,    -1,    91,    -1,    53,    -1,    54,    -1,    55,    -1,
      56,    -1,    57,    -1,    58,    -1,    59,    -1,    60,    -1,
      61,    -1,    62,    -1,    63,    -1,    64,    -1,    65,    -1,
      66,    -1,    67,    -1,    68,    -1,    69,    -1,    70,    -1,
     164,    -1,   164,    51,   163,    -1,    28,    -1,    78,    28,
      -1,   168,    -1,   169,    -1,   167,    -1,   165,    -1,   166,
      -1,    28,    75,    29,    -1,    80,    28,    51,    28,    79,
      -1,    80,    28,    51,    28,    51,    28,    79,    -1,    80,
      28,    51,    28,    51,    28,    51,    28,    79,    -1,    71,
      28,    51,    80,    28,    51,    28,    51,    28,    51,    28,
      79,    72,    -1,    74,    83,    -1,    74,    -1,    71,   170,
      72,    -1,    29,    -1,    30,    -1,    31,    -1,    28,    -1,
      28,    75,    29,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   203,   203,   204,   205,   206,   209,   209,   210,   210,
     213,   216,   216,   217,   217,   218,   221,   221,   222,   225,
     226,   229,   230,   232,   232,   233,   233,   235,   236,   237,
     238,   241,   242,   243,   244,   245,   246,   249,   250,   251,
     254,   256,   257,   259,   260,   272,   273,   276,   277,   279,
     280,   281,   282,   285,   287,   288,   289,   292,   293,   294,
     295,   296,   297,   298,   301,   302,   305,   306,   307,   310,
     311,   312,   313,   314,   315,   316,   317,   318,   319,   320,
     321,   322,   323,   324,   325,   328,   329,   331,   332,   334,
     335,   336,   338,   338,   339,   340,   341,   344,   344,   345,
     347,   348,   351,   352,   354,   355,   356,   357,   358,   359,
     360,   361,   362,   363,   364,   365,   366,   367,   368,   369,
     370,   371,   373,   374,   375,   376,   377,   378,   379,   380,
     381,   382,   385,   386,   388,   389,   390,   391,   394,   395,
     396,   397,   400,   401,   402,   403,   404,   405,   406,   407,
     408,   409,   410,   411,   412,   413,   414,   415,   416,   417,
     420,   421,   423,   424,   425,   426,   427,   428,   429,   430,
     433,   434,   435,   438,   445,   446,   449,   451,   452,   453,
     456,   457
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "STRING", "OPCODE", "ALIGN_DIRECTIVE",
  "BYTE_DIRECTIVE", "CONST_DIRECTIVE", "ENTRY_DIRECTIVE",
  "EXTERN_DIRECTIVE", "FILE_DIRECTIVE", "FUNC_DIRECTIVE",
  "GLOBAL_DIRECTIVE", "LOCAL_DIRECTIVE", "LOC_DIRECTIVE",
  "PARAM_DIRECTIVE", "REG_DIRECTIVE", "SECTION_DIRECTIVE",
  "SHARED_DIRECTIVE", "SREG_DIRECTIVE", "STRUCT_DIRECTIVE",
  "SURF_DIRECTIVE", "TARGET_DIRECTIVE", "TEX_DIRECTIVE", "UNION_DIRECTIVE",
  "VERSION_DIRECTIVE", "VISIBLE_DIRECTIVE", "MAXNTID_DIRECTIVE",
  "IDENTIFIER", "INT_OPERAND", "FLOAT_OPERAND", "DOUBLE_OPERAND",
  "S8_TYPE", "S16_TYPE", "S32_TYPE", "S64_TYPE", "U8_TYPE", "U16_TYPE",
  "U32_TYPE", "U64_TYPE", "F16_TYPE", "F32_TYPE", "F64_TYPE", "B8_TYPE",
  "B16_TYPE", "B32_TYPE", "B64_TYPE", "PRED_TYPE", "V2_TYPE", "V3_TYPE",
  "V4_TYPE", "COMMA", "PRED", "EQ_OPTION", "NE_OPTION", "LT_OPTION",
  "LE_OPTION", "GT_OPTION", "GE_OPTION", "LO_OPTION", "LS_OPTION",
  "HI_OPTION", "HS_OPTION", "EQU_OPTION", "NEU_OPTION", "LTU_OPTION",
  "LEU_OPTION", "GTU_OPTION", "GEU_OPTION", "NUM_OPTION", "NAN_OPTION",
  "LEFT_SQUARE_BRACKET", "RIGHT_SQUARE_BRACKET", "WIDE_OPTION",
  "SPECIAL_REGISTER", "PLUS", "COLON", "SEMI_COLON", "EXCLAMATION",
  "RIGHT_BRACE", "LEFT_BRACE", "EQUALS", "PERIOD", "DIMENSION_MODIFIER",
  "RN_OPTION", "RZ_OPTION", "RM_OPTION", "RP_OPTION", "RNI_OPTION",
  "RZI_OPTION", "RMI_OPTION", "RPI_OPTION", "UNI_OPTION",
  "GEOM_MODIFIER_1D", "GEOM_MODIFIER_2D", "GEOM_MODIFIER_3D", "SAT_OPTION",
  "FTZ_OPTION", "ATOMIC_AND", "ATOMIC_OR", "ATOMIC_XOR", "ATOMIC_CAS",
  "ATOMIC_EXCH", "ATOMIC_ADD", "ATOMIC_INC", "ATOMIC_DEC", "ATOMIC_MIN",
  "ATOMIC_MAX", "LEFT_ANGLE_BRACKET", "RIGHT_ANGLE_BRACKET", "LEFT_PAREN",
  "RIGHT_PAREN", "APPROX_OPTION", "FULL_OPTION", "ANY_OPTION",
  "ALL_OPTION", "GLOBAL_OPTION", "CTA_OPTION", "$accept", "input",
  "function_defn", "$@1", "$@2", "block_spec", "function_decl", "$@3",
  "$@4", "function_ident_param", "$@5", "function_decl_header",
  "param_list", "param_entry", "$@6", "$@7", "statement_list",
  "directive_statement", "variable_declaration", "variable_spec",
  "identifier_list", "identifier_spec", "var_spec_list", "var_spec",
  "align_spec", "space_spec", "addressable_spec", "type_spec",
  "vector_spec", "scalar_type", "initializer_list", "literal_list",
  "instruction_statement", "instruction", "$@8", "opcode_spec", "$@9",
  "pred_spec", "option_list", "option", "atomic_operation_spec",
  "rounding_mode", "floating_point_rounding_mode", "integer_rounding_mode",
  "compare_spec", "operand_list", "operand", "vector_operand",
  "tex_operand", "builtin_operand", "memory_operand", "literal_operand",
  "address_expression", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   350,   351,   352,   353,   354,
     355,   356,   357,   358,   359,   360,   361,   362,   363,   364,
     365,   366,   367,   368,   369,   370,   371,   372
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,   118,   119,   119,   119,   119,   121,   120,   122,   120,
     123,   125,   124,   126,   124,   124,   128,   127,   127,   129,
     129,   130,   130,   132,   131,   133,   131,   134,   134,   134,
     134,   135,   135,   135,   135,   135,   135,   136,   136,   136,
     137,   138,   138,   139,   139,   139,   139,   140,   140,   141,
     141,   141,   141,   142,   143,   143,   143,   144,   144,   144,
     144,   144,   144,   144,   145,   145,   146,   146,   146,   147,
     147,   147,   147,   147,   147,   147,   147,   147,   147,   147,
     147,   147,   147,   147,   147,   148,   148,   149,   149,   150,
     150,   150,   152,   151,   151,   151,   151,   154,   153,   153,
     155,   155,   156,   156,   157,   157,   157,   157,   157,   157,
     157,   157,   157,   157,   157,   157,   157,   157,   157,   157,
     157,   157,   158,   158,   158,   158,   158,   158,   158,   158,
     158,   158,   159,   159,   160,   160,   160,   160,   161,   161,
     161,   161,   162,   162,   162,   162,   162,   162,   162,   162,
     162,   162,   162,   162,   162,   162,   162,   162,   162,   162,
     163,   163,   164,   164,   164,   164,   164,   164,   164,   164,
     165,   165,   165,   166,   167,   167,   168,   169,   169,   169,
     170,   170
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     2,     2,     2,     0,     5,     0,     6,
       6,     0,     6,     0,     3,     1,     0,     5,     1,     1,
       1,     1,     3,     0,     4,     0,     4,     1,     1,     2,
       2,     2,     2,     4,     2,     3,     4,     2,     4,     4,
       1,     1,     3,     1,     4,     3,     4,     1,     2,     1,
       1,     1,     1,     2,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     2,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     3,     3,     1,     3,     2,
       2,     3,     0,    11,     6,     2,     1,     0,     3,     1,
       2,     3,     1,     2,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     1,     2,     1,     1,     1,     1,     1,     3,
       5,     7,     9,    13,     2,     1,     3,     1,     1,     1,
       1,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     0,     1,     0,    57,    19,    52,     0,    20,    58,
      59,     0,    60,    54,    61,    55,    62,     0,    63,     0,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    66,    67,    68,     4,
       5,    15,     3,     0,     0,    40,    47,    51,    49,    56,
      50,     0,    64,    53,     0,     0,    34,    32,     0,     0,
      11,     0,    31,    43,    37,    41,    48,    65,    35,     0,
       0,     0,     0,     0,     0,    18,    14,     0,     0,     0,
       0,    36,    33,    97,     0,     0,     0,    27,    28,     0,
      96,     0,     0,     0,    23,    25,     0,     0,     0,    45,
       0,    42,   177,   178,   179,     0,    38,    39,     0,    90,
     100,     0,     7,    29,    30,    89,   162,     0,   175,     0,
       0,     0,    95,   160,   167,   168,   166,   164,   165,     0,
       0,     0,     0,     0,     0,     0,    46,    44,     0,     0,
      87,   142,   143,   144,   145,   146,   147,   148,   149,   150,
     151,   152,   153,   154,   155,   156,   157,   158,   159,   109,
     134,   135,   136,   137,   138,   139,   140,   141,   108,   114,
     115,   116,   117,   118,   122,   123,   124,   125,   126,   127,
     128,   129,   130,   131,   119,   120,   110,   111,   112,   113,
     106,   104,    98,   102,   121,   107,   132,   133,   105,   101,
       0,   180,     0,   174,   163,     0,     0,     0,    91,     0,
       9,     0,     0,    12,     0,    21,    86,    85,     0,   103,
     169,     0,     0,   176,     0,    92,     0,   161,   160,     0,
      24,    26,     0,    17,    88,     0,   181,     0,     0,     0,
       0,    10,    22,     0,     0,   170,     0,    94,     0,     0,
       0,     0,     0,   171,     0,     0,     0,     0,     0,   172,
       0,     0,    93,     0,     0,   173
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     1,    39,    58,    59,    73,    40,    74,    61,    76,
      97,    41,   214,    96,   132,   133,    86,    87,    43,    44,
      64,    65,    45,    46,    47,    48,    49,    50,    51,    52,
     106,   139,    88,    89,   238,    90,   108,    91,   192,   193,
     194,   195,   196,   197,   198,   227,   228,   124,   125,   126,
     127,   128,   202
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -120
static const yytype_int16 yypact[] =
{
    -120,   347,  -120,    -4,  -120,  -120,  -120,     1,  -120,  -120,
    -120,     7,  -120,  -120,  -120,  -120,  -120,    57,  -120,    97,
    -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,
    -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,
      -3,   -24,  -120,   -42,   103,   440,  -120,  -120,  -120,  -120,
    -120,   136,  -120,  -120,   131,   106,    94,  -120,    70,   124,
    -120,   126,  -120,    16,   101,    75,  -120,  -120,  -120,   129,
     132,   394,   130,    81,    -6,    53,  -120,     4,   135,   103,
     -23,  -120,  -120,    52,    89,     6,   197,  -120,  -120,   108,
     113,   162,   137,   394,  -120,  -120,    79,    84,   114,  -120,
      86,  -120,  -120,  -120,  -120,   -23,  -120,  -120,     5,  -120,
    -120,   169,  -120,  -120,  -120,  -120,   125,   171,   120,   177,
     180,   271,  -120,   163,  -120,  -120,  -120,  -120,  -120,   140,
     192,   246,   440,   440,   126,    -6,  -120,  -120,   145,   147,
     176,  -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,
    -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,
    -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,
    -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,
    -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,
    -120,  -120,  -120,     5,  -120,  -120,  -120,  -120,  -120,  -120,
     219,    64,   182,  -120,  -120,   201,   146,   118,  -120,   212,
    -120,   103,   103,  -120,   -32,  -120,  -120,  -120,   -16,  -120,
    -120,   186,   241,  -120,   244,  -120,   271,  -120,   222,   248,
    -120,  -120,    -6,  -120,  -120,   247,  -120,    74,   252,   193,
     271,  -120,  -120,   254,   269,  -120,   271,  -120,   278,    76,
     256,   257,   281,  -120,   200,   283,   233,   271,   262,  -120,
     203,   287,  -120,   237,   245,  -120
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,  -120,   184,
    -120,  -120,  -120,  -119,  -120,  -120,   226,     2,  -120,  -111,
    -120,   -74,  -120,   275,  -120,  -120,   -79,   -77,  -120,   270,
     217,   105,   -75,   235,  -120,  -120,  -120,  -120,   134,  -120,
    -120,  -120,  -120,  -120,  -120,   -90,   -89,  -120,  -120,  -120,
    -120,   -78,  -120
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -100
static const yytype_int16 yytable[] =
{
     122,   123,   107,    42,   -13,   101,   102,   103,   104,    94,
      95,   114,     4,   102,   103,   104,   215,     9,    10,   232,
      12,   211,   212,    14,    -8,    53,    16,   140,    18,   190,
      54,   191,   206,    98,   110,    62,    55,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,   114,   105,   141,   142,
     143,   144,   145,   146,   147,   148,   149,   150,   151,   152,
     153,   154,   155,   156,   157,   158,    99,    -6,   159,   233,
     -99,   -99,   -99,   -99,   111,    56,    60,    77,   113,   160,
     161,   162,   163,   164,   165,   166,   167,   168,   169,   170,
     171,   172,   173,   174,   175,   176,   177,   178,   179,   180,
     181,   182,   183,   242,   190,   221,   191,   184,   185,   186,
     187,   188,   189,   -99,    78,   244,   -99,   252,    57,   -99,
     -99,    63,   -99,   113,    68,    69,   239,   230,   231,   222,
     140,   116,   102,   103,   104,    70,   116,   102,   103,   104,
      71,    72,    79,   245,    75,   253,    80,   250,    81,    92,
      82,    93,   -99,   -16,   100,   109,    83,   260,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,   117,   115,   136,   118,   130,   117,
     134,   119,   118,   120,   135,   137,   119,   199,   120,   201,
     200,    83,     3,   203,     4,   204,     6,     7,   205,     9,
      10,    11,    12,    13,   207,    14,    15,   208,    16,    17,
      18,   209,    19,   121,   216,    84,   217,   218,   226,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,   220,    85,
      83,     3,   224,     4,   223,     6,     7,   225,     9,    10,
      11,    12,    13,   229,    14,    15,   235,    16,    17,    18,
     236,    19,   237,   240,    84,   243,   112,   241,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,   249,    85,   116,
     102,   103,   104,   246,   247,   248,   251,   254,   255,   256,
     257,   258,   259,   261,   262,   263,   264,   265,   213,   131,
      66,    67,   138,   234,     0,   210,   129,   219,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   117,     0,     0,   118,     0,     2,     0,   119,
       0,   120,     3,     0,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,     0,    14,    15,     0,    16,    17,
      18,     0,    19,     0,     0,     0,     0,     0,     0,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    83,     3,
       0,     4,     0,     6,     7,     0,     9,    10,    11,    12,
      13,     0,    14,    15,     0,    16,    17,    18,     0,    19,
       0,     0,    84,     0,     0,     0,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    37,    38,     3,    85,     4,     0,     6,
       0,     0,     9,    10,     0,    12,    13,     0,    14,    15,
       0,    16,     0,    18,     0,     0,     0,     0,     0,     0,
       0,     0,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    37,
      38
};

static const yytype_int16 yycheck[] =
{
      90,    90,    80,     1,    28,    79,    29,    30,    31,    15,
      16,    86,     7,    29,    30,    31,   135,    12,    13,    51,
      15,   132,   133,    18,    27,    29,    21,   105,    23,   108,
      29,   108,   121,    29,    28,    77,    29,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,   131,    80,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    72,    80,    73,   111,
      28,    29,    30,    31,    78,    28,   110,    71,    86,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98,    99,   100,   101,   102,   103,   104,
     105,   106,   107,   232,   193,    51,   193,   112,   113,   114,
     115,   116,   117,    71,   108,    51,    74,    51,    31,    77,
      78,    28,    80,   131,     3,    29,   226,   211,   212,    75,
     218,    28,    29,    30,    31,    51,    28,    29,    30,    31,
      80,    27,    51,    79,    28,    79,    81,   246,    29,    29,
      28,    80,   110,   110,    29,    76,     4,   257,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    71,    77,    72,    74,    51,    71,
     111,    78,    74,    80,   110,   109,    78,    28,    80,    28,
      75,     4,     5,    83,     7,    28,     9,    10,    28,    12,
      13,    14,    15,    16,    51,    18,    19,    77,    21,    22,
      23,    29,    25,   110,    79,    28,    79,    51,   110,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    29,    52,
       4,     5,    51,     7,    72,     9,    10,   111,    12,    13,
      14,    15,    16,    51,    18,    19,    80,    21,    22,    23,
      29,    25,    28,    51,    28,    28,    79,    29,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    28,    52,    28,
      29,    30,    31,    51,   111,    51,    28,    51,    51,    28,
     110,    28,    79,    51,   111,    28,    79,    72,   134,    93,
      45,    51,   105,   218,    -1,    79,    91,   193,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    71,    -1,    -1,    74,    -1,     0,    -1,    78,
      -1,    80,     5,    -1,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    -1,    18,    19,    -1,    21,    22,
      23,    -1,    25,    -1,    -1,    -1,    -1,    -1,    -1,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,    50,     4,     5,
      -1,     7,    -1,     9,    10,    -1,    12,    13,    14,    15,
      16,    -1,    18,    19,    -1,    21,    22,    23,    -1,    25,
      -1,    -1,    28,    -1,    -1,    -1,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,     5,    52,     7,    -1,     9,
      -1,    -1,    12,    13,    -1,    15,    16,    -1,    18,    19,
      -1,    21,    -1,    23,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    32,    33,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,   119,     0,     5,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    18,    19,    21,    22,    23,    25,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,   120,
     124,   129,   135,   136,   137,   140,   141,   142,   143,   144,
     145,   146,   147,    29,    29,    29,    28,    31,   121,   122,
     110,   126,    77,    28,   138,   139,   141,   147,     3,    29,
      51,    80,    27,   123,   125,    28,   127,    71,   108,    51,
      81,    29,    28,     4,    28,    52,   134,   135,   150,   151,
     153,   155,    29,    80,    15,    16,   131,   128,    29,    72,
      29,   139,    29,    30,    31,    80,   148,   169,   154,    76,
      28,    78,    79,   135,   150,    77,    28,    71,    74,    78,
      80,   110,   163,   164,   165,   166,   167,   168,   169,   151,
      51,   134,   132,   133,   111,   110,    72,   109,   148,   149,
     169,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      62,    63,    64,    65,    66,    67,    68,    69,    70,    73,
      84,    85,    86,    87,    88,    89,    90,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,   107,   112,   113,   114,   115,   116,   117,
     144,   145,   156,   157,   158,   159,   160,   161,   162,    28,
      75,    28,   170,    83,    28,    28,   164,    51,    77,    29,
      79,   137,   137,   127,   130,   131,    79,    79,    51,   156,
      29,    51,    75,    72,    51,   111,   110,   163,   164,    51,
     139,   139,    51,   111,   149,    80,    29,    28,   152,   163,
      51,    29,   131,    28,    51,    79,    51,   111,    51,    28,
     164,    28,    51,    79,    51,    51,    28,   110,    28,    79,
     163,    51,   111,    28,    79,    72
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
#else
static void
yy_stack_print (yybottom, yytop)
    yytype_int16 *yybottom;
    yytype_int16 *yytop;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}

/* Prevent warnings from -Wmissing-prototypes.  */
#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */


/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*-------------------------.
| yyparse or yypush_parse.  |
`-------------------------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{


    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       `yyss': related to states.
       `yyvs': related to semantic values.

       Refer to the stacks thru separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yytoken = 0;
  yyss = yyssa;
  yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */
  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;

	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),
		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss_alloc, yyss);
	YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 6:

/* Line 1455 of yacc.c  */
#line 209 "ptx.y"
    { set_symtab((yyvsp[(1) - (1)].ptr_value)); ;}
    break;

  case 7:

/* Line 1455 of yacc.c  */
#line 209 "ptx.y"
    { end_function(); ;}
    break;

  case 8:

/* Line 1455 of yacc.c  */
#line 210 "ptx.y"
    { set_symtab((yyvsp[(1) - (1)].ptr_value)); ;}
    break;

  case 9:

/* Line 1455 of yacc.c  */
#line 210 "ptx.y"
    { end_function(); ;}
    break;

  case 11:

/* Line 1455 of yacc.c  */
#line 216 "ptx.y"
    { start_function((yyvsp[(1) - (2)].int_value)); ;}
    break;

  case 12:

/* Line 1455 of yacc.c  */
#line 216 "ptx.y"
    { (yyval.ptr_value) = reset_symtab(); ;}
    break;

  case 13:

/* Line 1455 of yacc.c  */
#line 217 "ptx.y"
    { start_function((yyvsp[(1) - (1)].int_value)); ;}
    break;

  case 14:

/* Line 1455 of yacc.c  */
#line 217 "ptx.y"
    { (yyval.ptr_value) = reset_symtab(); ;}
    break;

  case 15:

/* Line 1455 of yacc.c  */
#line 218 "ptx.y"
    { start_function((yyvsp[(1) - (1)].int_value)); add_function_name(""); g_func_decl=0; (yyval.ptr_value) = reset_symtab(); ;}
    break;

  case 16:

/* Line 1455 of yacc.c  */
#line 221 "ptx.y"
    { add_function_name((yyvsp[(1) - (1)].string_value)); ;}
    break;

  case 17:

/* Line 1455 of yacc.c  */
#line 221 "ptx.y"
    { g_func_decl=0; ;}
    break;

  case 18:

/* Line 1455 of yacc.c  */
#line 222 "ptx.y"
    { add_function_name((yyvsp[(1) - (1)].string_value)); g_func_decl=0; ;}
    break;

  case 19:

/* Line 1455 of yacc.c  */
#line 225 "ptx.y"
    { (yyval.int_value) = 1; g_func_decl=1; ;}
    break;

  case 20:

/* Line 1455 of yacc.c  */
#line 226 "ptx.y"
    { (yyval.int_value) = 0; g_func_decl=1; ;}
    break;

  case 21:

/* Line 1455 of yacc.c  */
#line 229 "ptx.y"
    { add_directive(); ;}
    break;

  case 22:

/* Line 1455 of yacc.c  */
#line 230 "ptx.y"
    { add_directive(); ;}
    break;

  case 23:

/* Line 1455 of yacc.c  */
#line 232 "ptx.y"
    { add_space_spec(PARAM_DIRECTIVE); ;}
    break;

  case 24:

/* Line 1455 of yacc.c  */
#line 232 "ptx.y"
    { add_function_arg(); ;}
    break;

  case 25:

/* Line 1455 of yacc.c  */
#line 233 "ptx.y"
    { add_space_spec(REG_DIRECTIVE); ;}
    break;

  case 26:

/* Line 1455 of yacc.c  */
#line 233 "ptx.y"
    { add_function_arg(); ;}
    break;

  case 27:

/* Line 1455 of yacc.c  */
#line 235 "ptx.y"
    { add_directive(); ;}
    break;

  case 28:

/* Line 1455 of yacc.c  */
#line 236 "ptx.y"
    { add_instruction(); ;}
    break;

  case 29:

/* Line 1455 of yacc.c  */
#line 237 "ptx.y"
    { add_directive(); ;}
    break;

  case 30:

/* Line 1455 of yacc.c  */
#line 238 "ptx.y"
    { add_instruction(); ;}
    break;

  case 35:

/* Line 1455 of yacc.c  */
#line 245 "ptx.y"
    { add_file((yyvsp[(2) - (3)].int_value),(yyvsp[(3) - (3)].string_value)); ;}
    break;

  case 37:

/* Line 1455 of yacc.c  */
#line 249 "ptx.y"
    { add_variables(); ;}
    break;

  case 38:

/* Line 1455 of yacc.c  */
#line 250 "ptx.y"
    { add_variables(); ;}
    break;

  case 39:

/* Line 1455 of yacc.c  */
#line 251 "ptx.y"
    { add_variables(); ;}
    break;

  case 40:

/* Line 1455 of yacc.c  */
#line 254 "ptx.y"
    { set_variable_type(); ;}
    break;

  case 43:

/* Line 1455 of yacc.c  */
#line 259 "ptx.y"
    { add_identifier((yyvsp[(1) - (1)].string_value),0,NON_ARRAY_IDENTIFIER); ;}
    break;

  case 44:

/* Line 1455 of yacc.c  */
#line 260 "ptx.y"
    {
		int i,lbase,l;
		char *id = NULL;
		lbase = strlen((yyvsp[(1) - (4)].string_value));
		for( i=0; i < (yyvsp[(3) - (4)].int_value); i++ ) { 
			l = lbase + (int)log10(i+1)+10;
			id = malloc(l);
			snprintf(id,l,"%s%u",(yyvsp[(1) - (4)].string_value),i);
			add_identifier(id,0,NON_ARRAY_IDENTIFIER); 
		}
		free((yyvsp[(1) - (4)].string_value));
	;}
    break;

  case 45:

/* Line 1455 of yacc.c  */
#line 272 "ptx.y"
    { add_identifier((yyvsp[(1) - (3)].string_value),0,ARRAY_IDENTIFIER_NO_DIM); ;}
    break;

  case 46:

/* Line 1455 of yacc.c  */
#line 273 "ptx.y"
    { add_identifier((yyvsp[(1) - (4)].string_value),(yyvsp[(3) - (4)].int_value),ARRAY_IDENTIFIER); ;}
    break;

  case 52:

/* Line 1455 of yacc.c  */
#line 282 "ptx.y"
    { add_extern_spec(); ;}
    break;

  case 53:

/* Line 1455 of yacc.c  */
#line 285 "ptx.y"
    { add_alignment_spec((yyvsp[(2) - (2)].int_value)); ;}
    break;

  case 54:

/* Line 1455 of yacc.c  */
#line 287 "ptx.y"
    {  add_space_spec(REG_DIRECTIVE); ;}
    break;

  case 55:

/* Line 1455 of yacc.c  */
#line 288 "ptx.y"
    {  add_space_spec(SREG_DIRECTIVE); ;}
    break;

  case 57:

/* Line 1455 of yacc.c  */
#line 292 "ptx.y"
    {  add_space_spec(CONST_DIRECTIVE); ;}
    break;

  case 58:

/* Line 1455 of yacc.c  */
#line 293 "ptx.y"
    {  add_space_spec(GLOBAL_DIRECTIVE); ;}
    break;

  case 59:

/* Line 1455 of yacc.c  */
#line 294 "ptx.y"
    {  add_space_spec(LOCAL_DIRECTIVE); ;}
    break;

  case 60:

/* Line 1455 of yacc.c  */
#line 295 "ptx.y"
    {  add_space_spec(PARAM_DIRECTIVE); ;}
    break;

  case 61:

/* Line 1455 of yacc.c  */
#line 296 "ptx.y"
    {  add_space_spec(SHARED_DIRECTIVE); ;}
    break;

  case 62:

/* Line 1455 of yacc.c  */
#line 297 "ptx.y"
    {  add_space_spec(SURF_DIRECTIVE); ;}
    break;

  case 63:

/* Line 1455 of yacc.c  */
#line 298 "ptx.y"
    {  add_space_spec(TEX_DIRECTIVE); ;}
    break;

  case 66:

/* Line 1455 of yacc.c  */
#line 305 "ptx.y"
    {  add_option(V2_TYPE); ;}
    break;

  case 67:

/* Line 1455 of yacc.c  */
#line 306 "ptx.y"
    {  add_option(V3_TYPE); ;}
    break;

  case 68:

/* Line 1455 of yacc.c  */
#line 307 "ptx.y"
    {  add_option(V4_TYPE); ;}
    break;

  case 69:

/* Line 1455 of yacc.c  */
#line 310 "ptx.y"
    { add_scalar_type_spec( S8_TYPE );  ;}
    break;

  case 70:

/* Line 1455 of yacc.c  */
#line 311 "ptx.y"
    { add_scalar_type_spec( S16_TYPE ); ;}
    break;

  case 71:

/* Line 1455 of yacc.c  */
#line 312 "ptx.y"
    { add_scalar_type_spec( S32_TYPE ); ;}
    break;

  case 72:

/* Line 1455 of yacc.c  */
#line 313 "ptx.y"
    { add_scalar_type_spec( S64_TYPE ); ;}
    break;

  case 73:

/* Line 1455 of yacc.c  */
#line 314 "ptx.y"
    { add_scalar_type_spec( U8_TYPE );  ;}
    break;

  case 74:

/* Line 1455 of yacc.c  */
#line 315 "ptx.y"
    { add_scalar_type_spec( U16_TYPE ); ;}
    break;

  case 75:

/* Line 1455 of yacc.c  */
#line 316 "ptx.y"
    { add_scalar_type_spec( U32_TYPE ); ;}
    break;

  case 76:

/* Line 1455 of yacc.c  */
#line 317 "ptx.y"
    { add_scalar_type_spec( U64_TYPE ); ;}
    break;

  case 77:

/* Line 1455 of yacc.c  */
#line 318 "ptx.y"
    { add_scalar_type_spec( F16_TYPE ); ;}
    break;

  case 78:

/* Line 1455 of yacc.c  */
#line 319 "ptx.y"
    { add_scalar_type_spec( F32_TYPE ); ;}
    break;

  case 79:

/* Line 1455 of yacc.c  */
#line 320 "ptx.y"
    { add_scalar_type_spec( F64_TYPE ); ;}
    break;

  case 80:

/* Line 1455 of yacc.c  */
#line 321 "ptx.y"
    { add_scalar_type_spec( B8_TYPE );  ;}
    break;

  case 81:

/* Line 1455 of yacc.c  */
#line 322 "ptx.y"
    { add_scalar_type_spec( B16_TYPE ); ;}
    break;

  case 82:

/* Line 1455 of yacc.c  */
#line 323 "ptx.y"
    { add_scalar_type_spec( B32_TYPE ); ;}
    break;

  case 83:

/* Line 1455 of yacc.c  */
#line 324 "ptx.y"
    { add_scalar_type_spec( B64_TYPE ); ;}
    break;

  case 84:

/* Line 1455 of yacc.c  */
#line 325 "ptx.y"
    { add_scalar_type_spec( PRED_TYPE ); ;}
    break;

  case 85:

/* Line 1455 of yacc.c  */
#line 328 "ptx.y"
    { add_array_initializer(); ;}
    break;

  case 86:

/* Line 1455 of yacc.c  */
#line 329 "ptx.y"
    { syntax_not_implemented(); ;}
    break;

  case 90:

/* Line 1455 of yacc.c  */
#line 335 "ptx.y"
    { add_label((yyvsp[(1) - (2)].string_value)); ;}
    break;

  case 92:

/* Line 1455 of yacc.c  */
#line 338 "ptx.y"
    { set_return(); ;}
    break;

  case 97:

/* Line 1455 of yacc.c  */
#line 344 "ptx.y"
    { add_opcode((yyvsp[(1) - (1)].int_value)); ;}
    break;

  case 99:

/* Line 1455 of yacc.c  */
#line 345 "ptx.y"
    { add_opcode((yyvsp[(1) - (1)].int_value)); ;}
    break;

  case 100:

/* Line 1455 of yacc.c  */
#line 347 "ptx.y"
    { add_pred((yyvsp[(2) - (2)].string_value),0); ;}
    break;

  case 101:

/* Line 1455 of yacc.c  */
#line 348 "ptx.y"
    { add_pred((yyvsp[(3) - (3)].string_value),1); ;}
    break;

  case 108:

/* Line 1455 of yacc.c  */
#line 358 "ptx.y"
    { add_option(UNI_OPTION); ;}
    break;

  case 109:

/* Line 1455 of yacc.c  */
#line 359 "ptx.y"
    { add_option(WIDE_OPTION); ;}
    break;

  case 110:

/* Line 1455 of yacc.c  */
#line 360 "ptx.y"
    { add_option(ANY_OPTION); ;}
    break;

  case 111:

/* Line 1455 of yacc.c  */
#line 361 "ptx.y"
    { add_option(ALL_OPTION); ;}
    break;

  case 112:

/* Line 1455 of yacc.c  */
#line 362 "ptx.y"
    { add_option(GLOBAL_OPTION); ;}
    break;

  case 113:

/* Line 1455 of yacc.c  */
#line 363 "ptx.y"
    { add_option(CTA_OPTION); ;}
    break;

  case 114:

/* Line 1455 of yacc.c  */
#line 364 "ptx.y"
    { add_option(GEOM_MODIFIER_1D); ;}
    break;

  case 115:

/* Line 1455 of yacc.c  */
#line 365 "ptx.y"
    { add_option(GEOM_MODIFIER_2D); ;}
    break;

  case 116:

/* Line 1455 of yacc.c  */
#line 366 "ptx.y"
    { add_option(GEOM_MODIFIER_3D); ;}
    break;

  case 117:

/* Line 1455 of yacc.c  */
#line 367 "ptx.y"
    { add_option(SAT_OPTION); ;}
    break;

  case 118:

/* Line 1455 of yacc.c  */
#line 368 "ptx.y"
    { add_option(FTZ_OPTION); ;}
    break;

  case 119:

/* Line 1455 of yacc.c  */
#line 369 "ptx.y"
    { add_option(APPROX_OPTION); ;}
    break;

  case 120:

/* Line 1455 of yacc.c  */
#line 370 "ptx.y"
    { add_option(FULL_OPTION); ;}
    break;

  case 122:

/* Line 1455 of yacc.c  */
#line 373 "ptx.y"
    { add_option(ATOMIC_AND); ;}
    break;

  case 123:

/* Line 1455 of yacc.c  */
#line 374 "ptx.y"
    { add_option(ATOMIC_OR); ;}
    break;

  case 124:

/* Line 1455 of yacc.c  */
#line 375 "ptx.y"
    { add_option(ATOMIC_XOR); ;}
    break;

  case 125:

/* Line 1455 of yacc.c  */
#line 376 "ptx.y"
    { add_option(ATOMIC_CAS); ;}
    break;

  case 126:

/* Line 1455 of yacc.c  */
#line 377 "ptx.y"
    { add_option(ATOMIC_EXCH); ;}
    break;

  case 127:

/* Line 1455 of yacc.c  */
#line 378 "ptx.y"
    { add_option(ATOMIC_ADD); ;}
    break;

  case 128:

/* Line 1455 of yacc.c  */
#line 379 "ptx.y"
    { add_option(ATOMIC_INC); ;}
    break;

  case 129:

/* Line 1455 of yacc.c  */
#line 380 "ptx.y"
    { add_option(ATOMIC_DEC); ;}
    break;

  case 130:

/* Line 1455 of yacc.c  */
#line 381 "ptx.y"
    { add_option(ATOMIC_MIN); ;}
    break;

  case 131:

/* Line 1455 of yacc.c  */
#line 382 "ptx.y"
    { add_option(ATOMIC_MAX); ;}
    break;

  case 134:

/* Line 1455 of yacc.c  */
#line 388 "ptx.y"
    { add_option(RN_OPTION); ;}
    break;

  case 135:

/* Line 1455 of yacc.c  */
#line 389 "ptx.y"
    { add_option(RZ_OPTION); ;}
    break;

  case 136:

/* Line 1455 of yacc.c  */
#line 390 "ptx.y"
    { add_option(RM_OPTION); ;}
    break;

  case 137:

/* Line 1455 of yacc.c  */
#line 391 "ptx.y"
    { add_option(RP_OPTION); ;}
    break;

  case 138:

/* Line 1455 of yacc.c  */
#line 394 "ptx.y"
    { add_option(RNI_OPTION); ;}
    break;

  case 139:

/* Line 1455 of yacc.c  */
#line 395 "ptx.y"
    { add_option(RZI_OPTION); ;}
    break;

  case 140:

/* Line 1455 of yacc.c  */
#line 396 "ptx.y"
    { add_option(RMI_OPTION); ;}
    break;

  case 141:

/* Line 1455 of yacc.c  */
#line 397 "ptx.y"
    { add_option(RPI_OPTION); ;}
    break;

  case 142:

/* Line 1455 of yacc.c  */
#line 400 "ptx.y"
    { add_option(EQ_OPTION); ;}
    break;

  case 143:

/* Line 1455 of yacc.c  */
#line 401 "ptx.y"
    { add_option(NE_OPTION); ;}
    break;

  case 144:

/* Line 1455 of yacc.c  */
#line 402 "ptx.y"
    { add_option(LT_OPTION); ;}
    break;

  case 145:

/* Line 1455 of yacc.c  */
#line 403 "ptx.y"
    { add_option(LE_OPTION); ;}
    break;

  case 146:

/* Line 1455 of yacc.c  */
#line 404 "ptx.y"
    { add_option(GT_OPTION); ;}
    break;

  case 147:

/* Line 1455 of yacc.c  */
#line 405 "ptx.y"
    { add_option(GE_OPTION); ;}
    break;

  case 148:

/* Line 1455 of yacc.c  */
#line 406 "ptx.y"
    { add_option(LO_OPTION); ;}
    break;

  case 149:

/* Line 1455 of yacc.c  */
#line 407 "ptx.y"
    { add_option(LS_OPTION); ;}
    break;

  case 150:

/* Line 1455 of yacc.c  */
#line 408 "ptx.y"
    { add_option(HI_OPTION); ;}
    break;

  case 151:

/* Line 1455 of yacc.c  */
#line 409 "ptx.y"
    { add_option(HS_OPTION); ;}
    break;

  case 152:

/* Line 1455 of yacc.c  */
#line 410 "ptx.y"
    { add_option(EQU_OPTION); ;}
    break;

  case 153:

/* Line 1455 of yacc.c  */
#line 411 "ptx.y"
    { add_option(NEU_OPTION); ;}
    break;

  case 154:

/* Line 1455 of yacc.c  */
#line 412 "ptx.y"
    { add_option(LTU_OPTION); ;}
    break;

  case 155:

/* Line 1455 of yacc.c  */
#line 413 "ptx.y"
    { add_option(LEU_OPTION); ;}
    break;

  case 156:

/* Line 1455 of yacc.c  */
#line 414 "ptx.y"
    { add_option(GTU_OPTION); ;}
    break;

  case 157:

/* Line 1455 of yacc.c  */
#line 415 "ptx.y"
    { add_option(GEU_OPTION); ;}
    break;

  case 158:

/* Line 1455 of yacc.c  */
#line 416 "ptx.y"
    { add_option(NUM_OPTION); ;}
    break;

  case 159:

/* Line 1455 of yacc.c  */
#line 417 "ptx.y"
    { add_option(NAN_OPTION); ;}
    break;

  case 162:

/* Line 1455 of yacc.c  */
#line 423 "ptx.y"
    { add_scalar_operand( (yyvsp[(1) - (1)].string_value) ); ;}
    break;

  case 163:

/* Line 1455 of yacc.c  */
#line 424 "ptx.y"
    { add_neg_pred_operand( (yyvsp[(2) - (2)].string_value) ); ;}
    break;

  case 169:

/* Line 1455 of yacc.c  */
#line 430 "ptx.y"
    { add_address_operand((yyvsp[(1) - (3)].string_value),(yyvsp[(3) - (3)].int_value)); ;}
    break;

  case 170:

/* Line 1455 of yacc.c  */
#line 433 "ptx.y"
    { add_2vector_operand((yyvsp[(2) - (5)].string_value),(yyvsp[(4) - (5)].string_value)); ;}
    break;

  case 171:

/* Line 1455 of yacc.c  */
#line 434 "ptx.y"
    { add_3vector_operand((yyvsp[(2) - (7)].string_value),(yyvsp[(4) - (7)].string_value),(yyvsp[(6) - (7)].string_value)); ;}
    break;

  case 172:

/* Line 1455 of yacc.c  */
#line 435 "ptx.y"
    { add_4vector_operand((yyvsp[(2) - (9)].string_value),(yyvsp[(4) - (9)].string_value),(yyvsp[(6) - (9)].string_value),(yyvsp[(8) - (9)].string_value)); ;}
    break;

  case 173:

/* Line 1455 of yacc.c  */
#line 440 "ptx.y"
    { 
		add_scalar_operand((yyvsp[(2) - (13)].string_value));
		add_4vector_operand((yyvsp[(5) - (13)].string_value),(yyvsp[(7) - (13)].string_value),(yyvsp[(9) - (13)].string_value),(yyvsp[(11) - (13)].string_value)); 
	;}
    break;

  case 174:

/* Line 1455 of yacc.c  */
#line 445 "ptx.y"
    { add_builtin_operand((yyvsp[(1) - (2)].int_value),(yyvsp[(2) - (2)].int_value)); ;}
    break;

  case 175:

/* Line 1455 of yacc.c  */
#line 446 "ptx.y"
    { add_builtin_operand((yyvsp[(1) - (1)].int_value),-1); ;}
    break;

  case 176:

/* Line 1455 of yacc.c  */
#line 449 "ptx.y"
    { add_memory_operand(); ;}
    break;

  case 177:

/* Line 1455 of yacc.c  */
#line 451 "ptx.y"
    { add_literal_int((yyvsp[(1) - (1)].int_value)); ;}
    break;

  case 178:

/* Line 1455 of yacc.c  */
#line 452 "ptx.y"
    { add_literal_float((yyvsp[(1) - (1)].float_value)); ;}
    break;

  case 179:

/* Line 1455 of yacc.c  */
#line 453 "ptx.y"
    { add_literal_double((yyvsp[(1) - (1)].double_value)); ;}
    break;

  case 180:

/* Line 1455 of yacc.c  */
#line 456 "ptx.y"
    { add_address_operand((yyvsp[(1) - (1)].string_value),0); ;}
    break;

  case 181:

/* Line 1455 of yacc.c  */
#line 457 "ptx.y"
    { add_address_operand((yyvsp[(1) - (3)].string_value),(yyvsp[(3) - (3)].int_value)); ;}
    break;



/* Line 1455 of yacc.c  */
#line 2772 "ptx.tab.c"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined(yyoverflow) || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}



/* Line 1675 of yacc.c  */
#line 460 "ptx.y"


extern int ptx_lineno;
extern const char *g_filename;

void syntax_not_implemented()
{
	printf("Parse error (%s:%u): this syntax is not (yet) implemented:\n",g_filename,ptx_lineno);
	ptx_error(NULL);
	abort();
}

