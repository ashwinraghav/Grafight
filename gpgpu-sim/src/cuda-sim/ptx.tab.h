
/* A Bison parser, made by GNU Bison 2.4.1.  */

/* Skeleton interface for Bison's Yacc-like parsers in C
   
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

/* Line 1676 of yacc.c  */
#line 64 "ptx.y"

  double double_value;
  float  float_value;
  int    int_value;
  char * string_value;
  void * ptr_value;



/* Line 1676 of yacc.c  */
#line 179 "ptx.tab.h"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE ptx_lval;


